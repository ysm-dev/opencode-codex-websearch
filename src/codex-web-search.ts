import { randomUUID } from "node:crypto"
import { existsSync } from "node:fs"
import { readFile, realpath } from "node:fs/promises"
import { homedir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin/tool"

const SEARCH_ENDPOINT = "https://chatgpt.com/backend-api/codex/alpha/search"
const MODELS_ENDPOINT = "https://chatgpt.com/backend-api/codex/models?client_version=0.147.0"
const REQUEST_TIMEOUT_MS = 15_000
const DEFAULT_MAX_RESULTS = 8
const MAX_OUTPUT_BYTES = 20_000
const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, { granularity: "grapheme" })

type CodexAuth = {
  accessToken: string
  accountId?: string
  fedramp: boolean
  uploaded: boolean
}

type OpenCodeOAuth = {
  type: "oauth"
  access: string
  expires: number
  accountId?: string
  fedramp?: boolean
}

type SearchResult = {
  title: string
  url: string
  snippet?: string
}

const registeredClients = new WeakSet()
let codexModel: string | undefined

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function errorCode(error: unknown) {
  return isRecord(error) && typeof error.code === "string" ? error.code : undefined
}

function cleanText(value: unknown, maxBytes: number): string | undefined {
  if (typeof value !== "string") return undefined

  const text = value.replace(/\s+/g, " ").trim()
  if (!text) return undefined
  return truncateText(text, maxBytes)
}

function truncateText(value: string, maxBytes: number) {
  if (Buffer.byteLength(value, "utf8") <= maxBytes) return value

  const segments: string[] = []
  let size = Buffer.byteLength("...", "utf8")
  for (const item of GRAPHEME_SEGMENTER.segment(value)) {
    const next = Buffer.byteLength(item.segment, "utf8")
    if (size + next > maxBytes) break
    segments.push(item.segment)
    size += next
  }
  return `${segments.join("")}...`
}

function normalizeUrl(value: unknown): string | undefined {
  if (typeof value !== "string" || value.length > 2_048) return undefined

  try {
    const url = new URL(value)
    if (url.protocol !== "http:" && url.protocol !== "https:") return undefined
    const normalized = url.toString()
    return Buffer.byteLength(normalized, "utf8") <= 2_048 ? normalized : undefined
  } catch {
    return undefined
  }
}

function normalizeResponse(payload: unknown, limit: number) {
  if (
    !isRecord(payload) ||
    typeof payload.output !== "string" ||
    (payload.results !== undefined && payload.results !== null && !Array.isArray(payload.results))
  ) {
    throw new Error("Codex web search returned an invalid response")
  }

  const results: SearchResult[] = []
  for (const item of payload.results ?? []) {
    if (!isRecord(item)) continue

    const url = normalizeUrl(item.url)
    if (!url) continue

    results.push({
      title: cleanText(item.title, 300) ?? url,
      url,
      snippet: cleanText(item.snippet, 1_000),
    })
    if (results.length >= limit) break
  }

  return results
}

function formatResponse(query: string, results: SearchResult[]): string {
  const items: string[] = []
  for (const result of results) {
    const snippet = result.snippet ? `\n${result.snippet}` : ""
    const item = `## ${result.title}\n${result.url}${snippet}`
    if (items.length && Buffer.byteLength([...items, item].join("\n\n"), "utf8") > MAX_OUTPUT_BYTES) break
    items.push(item)
  }

  if (items.length === 0) return `No web search results found for: "${query}".`
  return truncateText(items.join("\n\n"), MAX_OUTPUT_BYTES)
}

async function loadCodexAuth() {
  const environment = openCodeAuthEnvironment()
  const auth = parseOpenCodeOAuth((environment ?? (await loadOpenCodeAuthFile())).openai)
  if (!auth) {
    throw new Error("OpenCode ChatGPT authentication is unavailable; run `opencode auth login` and retry")
  }
  if (auth.expires <= Date.now() + REQUEST_TIMEOUT_MS) {
    if (environment) {
      throw new Error("Uploaded OpenCode ChatGPT authentication has expired; recreate the workspace and retry")
    }
    throw new Error("OpenCode ChatGPT authentication has expired; run `opencode auth login` and retry")
  }
  return codexAuth(auth, environment !== undefined)
}

function parseOpenCodeOAuth(auth: unknown): OpenCodeOAuth | undefined {
  if (!isRecord(auth) || auth.type !== "oauth" || typeof auth.access !== "string" || typeof auth.expires !== "number") {
    return undefined
  }
  return {
    type: "oauth",
    access: auth.access,
    expires: auth.expires,
    ...(typeof auth.accountId === "string" ? { accountId: auth.accountId } : {}),
    ...(typeof auth.fedramp === "boolean" ? { fedramp: auth.fedramp } : {}),
  }
}

async function loadOpenCodeAuthFile() {
  const parsed = parseOpenCodeAuth(await readFile(openCodeAuthPath(), "utf8"))
  if (parsed) return parsed
  throw new Error("OpenCode authentication data is invalid")
}

function openCodeAuthEnvironment() {
  const environment = process.env.OPENCODE_AUTH_CONTENT?.trim()
  return environment ? parseOpenCodeAuth(environment) : undefined
}

function parseOpenCodeAuth(content: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(content)
    return isRecord(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

function openCodeAuthPath() {
  const dataRoot = process.env.XDG_DATA_HOME?.trim() || join(homedir(), ".local", "share")
  return join(dataRoot, "opencode", "auth.json")
}

function openCodeConfigPaths() {
  const xdgConfig = process.env.XDG_CONFIG_HOME?.trim() || join(homedir(), ".config")
  const custom = process.env.OPENCODE_CONFIG_DIR?.trim()
  return [
    resolve(join(xdgConfig, "opencode")),
    resolve(join(homedir(), ".opencode")),
    ...(custom ? [resolve(custom)] : []),
  ]
}

function projectConfigPaths(directory: string, worktree: string): string[] {
  const root = resolve(worktree)
  const current = resolve(directory)
  if (current === root) return [resolve(root, ".opencode")]
  const parent = dirname(current)
  if (parent === current) return [resolve(current, ".opencode")]
  return [resolve(current, ".opencode"), ...projectConfigPaths(parent, root)]
}

function canonicalPath(path: string) {
  return realpath(path).catch((error: unknown) => {
    if (errorCode(error) === "ENOENT" || errorCode(error) === "ENOTDIR") return resolve(path)
    throw error
  })
}

function codexAuth(auth: OpenCodeOAuth, uploaded: boolean): CodexAuth {
  const claims = tokenAuthClaims(auth.access)
  return {
    accessToken: auth.access,
    accountId: cleanText(auth.accountId, 1_000) ?? claims.accountId,
    fedramp: auth.fedramp ?? claims.fedramp,
    uploaded,
  }
}

function authHeaders(auth: CodexAuth) {
  const headers: Record<string, string> = {
    Accept: "application/json",
    Authorization: `Bearer ${auth.accessToken}`,
    "User-Agent": "codex-cli/0.147.0-alpha.6.5",
  }
  if (auth.accountId) headers["ChatGPT-Account-ID"] = auth.accountId
  if (auth.fedramp) headers["X-OpenAI-Fedramp"] = "true"
  return headers
}

function requireSuccessfulResponse(response: Response, auth: CodexAuth, operation: string) {
  if (response.status === 401) {
    if (auth.uploaded) {
      throw new Error(
        "Uploaded OpenCode ChatGPT authentication was rejected or expired; recreate the workspace with current credentials",
      )
    }
    throw new Error("OpenCode ChatGPT authentication was rejected or expired; run `opencode auth login` and retry")
  }
  if (response.status === 403 && response.headers.get("cf-mitigated")?.toLowerCase() === "challenge") {
    throw new Error(
      `${operation} was blocked by a Cloudflare browser challenge; retry later or from a different network`,
    )
  }
  if (response.status === 403) {
    throw new Error(
      `${operation} is forbidden for the current ChatGPT account, model, or workspace; verify that this account has Codex access`,
    )
  }
  if (response.status === 429) throw new Error(`${operation} rate limit exceeded; retry later`)
  if (!response.ok) throw new Error(`${operation} failed with HTTP ${response.status}`)
}

async function loadCodexModel(auth: CodexAuth, signal: AbortSignal) {
  if (codexModel) return codexModel
  codexModel = await discoverCodexModel(auth, signal)
  return codexModel
}

async function discoverCodexModel(auth: CodexAuth, signal: AbortSignal) {
  const response = await fetch(MODELS_ENDPOINT, { headers: authHeaders(auth), signal }).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : "unknown network error"
    throw new Error(`Codex model discovery request failed: ${message}`, { cause: error })
  })
  requireSuccessfulResponse(response, auth, "Codex model discovery")

  const payload: unknown = await response.json().catch((error: unknown) => {
    throw new Error("Codex model discovery returned invalid JSON", { cause: error })
  })
  if (!isRecord(payload) || !Array.isArray(payload.models)) {
    throw new Error("Codex model discovery returned an invalid response")
  }
  const candidates = payload.models
    .flatMap((model) => {
      if (
        !isRecord(model) ||
        typeof model.slug !== "string" ||
        !model.slug ||
        typeof model.priority !== "number" ||
        !Number.isFinite(model.priority) ||
        !["list", "hide", "none"].includes(String(model.visibility))
      ) {
        return []
      }
      return [{ slug: model.slug, priority: model.priority, visibility: String(model.visibility) }]
    })
    .sort((left, right) => left.priority - right.priority)
  const model = candidates.find((candidate) => candidate.visibility === "list") ?? candidates[0]
  if (!model) throw new Error("Could not determine an account-eligible Codex model for web search; retry later")
  return model.slug
}

function tokenAuthClaims(token: string) {
  const claims = parseJwtClaims(token)
  const nestedAuth = claims?.["https://api.openai.com/auth"]
  const nested = isRecord(nestedAuth) ? nestedAuth : undefined
  const organizations = Array.isArray(claims?.organizations) ? claims.organizations : []
  const organization = organizations.find(isRecord)
  return {
    accountId:
      cleanText(claims?.chatgpt_account_id, 1_000) ??
      cleanText(nested?.chatgpt_account_id, 1_000) ??
      cleanText(organization?.id, 1_000),
    fedramp:
      typeof claims?.chatgpt_account_is_fedramp === "boolean"
        ? claims.chatgpt_account_is_fedramp
        : nested?.chatgpt_account_is_fedramp === true,
  }
}

function parseJwtClaims(token: string): Record<string, unknown> | undefined {
  const payload = token.split(".")[1]
  if (!payload) return undefined

  try {
    const parsed: unknown = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"))
    return isRecord(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

export const CodexWebSearchPlugin: Plugin = async ({ client, directory, worktree }) => {
  const currentPath = await canonicalPath(fileURLToPath(import.meta.url))
  const projectConfigDisabled = ["1", "true"].includes(process.env.OPENCODE_DISABLE_PROJECT_CONFIG?.toLowerCase() ?? "")
  const projectRoots = projectConfigDisabled
    ? []
    : await Promise.all(projectConfigPaths(directory, worktree).map(canonicalPath))
  const projectPaths = await Promise.all(
    projectRoots.flatMap((root) =>
      ["plugin", "plugins"].map((pluginDirectory) =>
        canonicalPath(resolve(root, pluginDirectory, "codex-web-search.ts")),
      ),
    ),
  )
  const globalPaths = await Promise.all(
    openCodeConfigPaths().flatMap((root) =>
      ["plugin", "plugins"].map((pluginDirectory) => canonicalPath(join(root, pluginDirectory, "codex-web-search.ts"))),
    ),
  )
  const selectedProjectPath = projectPaths.find((path) => existsSync(path))
  const selectedGlobalPath = globalPaths.findLast((path) => existsSync(path))
  if (selectedProjectPath && selectedProjectPath !== currentPath) return {}
  if (!selectedProjectPath && globalPaths.includes(currentPath) && selectedGlobalPath !== currentPath) return {}
  if (registeredClients.has(client)) return {}
  registeredClients.add(client)

  return {
    tool: {
      codex_web_search: tool({
        description: "Search the web",
        args: {
          query: tool.schema.string().trim().min(1).max(500).describe("The web search query"),
          max_results: tool.schema
            .number()
            .int()
            .min(1)
            .max(20)
            .optional()
            .describe("Maximum number of structured results to return (default: 8)"),
          recency: tool.schema
            .number()
            .int()
            .min(1)
            .max(3_650)
            .optional()
            .describe("Only return results from the last N days"),
          domains: tool.schema
            .array(tool.schema.string().trim().min(1).max(253))
            .max(20)
            .optional()
            .describe("Only return results from these domains, such as github.com"),
        },
        async execute({ query, max_results, recency, domains }, context) {
          const maxResults = max_results ?? DEFAULT_MAX_RESULTS
          const searchQuery: { q: string; recency?: number; domains?: string[] } = { q: query }
          if (recency !== undefined) searchQuery.recency = recency
          if (domains?.length) searchQuery.domains = domains

          context.metadata({ title: `Web search: ${query}` })
          await context.ask({
            permission: "codex_web_search",
            patterns: [query],
            always: ["*"],
            metadata: { query, max_results, recency, domains, provider: "codex-standalone-search" },
          })

          const controller = new AbortController()
          let timedOut = false
          const cancelRequest = () => controller.abort()
          if (context.abort.aborted) cancelRequest()
          else context.abort.addEventListener("abort", cancelRequest, { once: true })
          const timeout = setTimeout(() => {
            timedOut = true
            controller.abort()
          }, REQUEST_TIMEOUT_MS)

          try {
            const auth = await loadCodexAuth()
            const model = await loadCodexModel(auth, controller.signal)
            const headers = { ...authHeaders(auth), "Content-Type": "application/json" }

            const response = await fetch(SEARCH_ENDPOINT, {
              method: "POST",
              headers,
              body: JSON.stringify({
                id: `search_session_${randomUUID().replaceAll("-", "").slice(0, 16)}`,
                model,
                commands: { search_query: [searchQuery] },
              }),
              signal: controller.signal,
            }).catch((error: unknown) => {
              const message = error instanceof Error ? error.message : "unknown network error"
              throw new Error(`Codex web search request failed: ${message}`, { cause: error })
            })

            requireSuccessfulResponse(response, auth, "Codex web search")

            const payload: unknown = await response.json().catch((error: unknown) => {
              throw new Error("Codex web search returned invalid JSON", { cause: error })
            })
            const results = normalizeResponse(payload, maxResults)
            return {
              title: `Web search: ${query}`,
              output: formatResponse(query, results),
              metadata: { query, resultCount: results.length },
            }
          } catch (error) {
            if (timedOut) {
              throw new Error(`Codex web search timed out after ${REQUEST_TIMEOUT_MS}ms`, { cause: error })
            }
            if (context.abort.aborted) throw new Error("Codex web search was cancelled", { cause: error })
            throw error
          } finally {
            clearTimeout(timeout)
            context.abort.removeEventListener("abort", cancelRequest)
          }
        },
      }),
    },
  }
}

export default CodexWebSearchPlugin
