import { randomUUID } from "node:crypto"
import { Plugin } from "@opencode/plugin"
import type { IntegrationDomain } from "@opencode/plugin/promise/integration"
import { HttpClientError, HttpClientRequest, HttpClientResponse } from "effect/unstable/http"

const SEARCH_ENDPOINT = "https://chatgpt.com/backend-api/codex/alpha/search"
const MODELS_ENDPOINT = "https://chatgpt.com/backend-api/codex/models?client_version=0.147.0"
const REQUEST_TIMEOUT_MS = 15_000
const DEFAULT_MAX_RESULTS = 8
const MAX_OUTPUT_BYTES = 20_000
const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, { granularity: "grapheme" })

type CodexAuth = {
  accessToken: string
  accountId?: string
  credentialID: string
  fedramp: boolean
}

type SearchOptions = {
  max_results: number
  recency?: number
  domains?: string[]
}

export const CodexWebSearchPlugin = Plugin.define({
  id: "opencode-codex-websearch",
  async setup(ctx) {
    const options = parseOptions(ctx.options)
    const lifetime = new AbortController()
    let cached: { credentialID: string; accountId?: string; model: string } | undefined

    await ctx.websearch.transform((editor) => {
      editor.add({
        id: "codex",
        name: "ChatGPT Codex",
        async execute({ query }, { signal }) {
          const q = query.trim()
          if (!q || q.length > 500) throw new Error("Codex search query must contain 1 to 500 characters")

          return withTimeout(AbortSignal.any([signal, lifetime.signal]), async (signal) => {
            const auth = await loadCodexAuth(ctx.integration.connection)
            signal.throwIfAborted()
            const model =
              cached?.credentialID === auth.credentialID && cached.accountId === auth.accountId
                ? cached.model
                : await discoverCodexModel(auth, signal)
            signal.throwIfAborted()
            cached = { credentialID: auth.credentialID, accountId: auth.accountId, model }

            const searchQuery = {
              q,
              ...(options.recency !== undefined ? { recency: options.recency } : {}),
              ...(options.domains?.length ? { domains: options.domains } : {}),
            }
            const payload = await requestJson(
              SEARCH_ENDPOINT,
              auth,
              signal,
              "Codex web search",
              {
                id: `search_session_${randomUUID().replaceAll("-", "").slice(0, 16)}`,
                model,
                commands: { search_query: [searchQuery] },
              },
            )
            return normalizeResponse(payload, options.max_results)
          })
        },
      })
    })

    return () => {
      lifetime.abort()
      cached = undefined
    }
  },
})

export default CodexWebSearchPlugin

function parseOptions(options: Record<string, unknown>): SearchOptions {
  for (const key of Object.keys(options)) {
    if (!["max_results", "recency", "domains"].includes(key)) throw new Error(`Unknown Codex search option: ${key}`)
  }
  const integer = (key: string, maximum: number) => {
    const value = options[key]
    if (value === undefined) return undefined
    if (typeof value !== "number" || !Number.isInteger(value) || value < 1 || value > maximum) {
      throw new Error(`Codex search option ${key} must be an integer from 1 to ${maximum}`)
    }
    return value
  }
  const max_results = integer("max_results", 20) ?? DEFAULT_MAX_RESULTS
  const recency = integer("recency", 3_650)
  const domains = options.domains
  if (domains === undefined) return { max_results, recency }
  if (!Array.isArray(domains) || domains.length > 20) {
    throw new Error("Codex search option domains must be an array of at most 20 domains")
  }
  return {
    max_results,
    recency,
    domains: domains.map((domain: unknown) => {
      if (typeof domain !== "string" || !domain.trim() || domain.trim().length > 253) {
        throw new Error("Each Codex search domain must contain 1 to 253 characters")
      }
      return domain.trim()
    }),
  }
}

async function loadCodexAuth(connectionAPI: IntegrationDomain["connection"]): Promise<CodexAuth> {
  const connection = await connectionAPI.active("openai")
  const credential = connection ? await connectionAPI.resolve(connection) : undefined
  if (
    connection?.type !== "credential" ||
    credential?.type !== "oauth" ||
    !["chatgpt-browser", "chatgpt-headless"].includes(credential.methodID) ||
    !credential.access
  ) {
    throw new Error("OpenCode ChatGPT authentication is unavailable; run `opencode auth login` and select ChatGPT")
  }
  // OpenCode resolves SQLite credentials and refreshes OAuth before returning them.
  if (credential.expires <= Date.now() + REQUEST_TIMEOUT_MS) {
    throw new Error("OpenCode ChatGPT authentication has expired; reconnect ChatGPT with `opencode auth login`")
  }
  const claims = tokenAuthClaims(credential.access)
  return {
    accessToken: credential.access,
    credentialID: connection.id,
    accountId: cleanText(credential.metadata?.accountID, 1_000) ?? claims.accountId,
    fedramp: typeof credential.metadata?.fedramp === "boolean" ? credential.metadata.fedramp : claims.fedramp,
  }
}

async function withTimeout<T>(parent: AbortSignal, work: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const controller = new AbortController()
  const signal = AbortSignal.any([parent, controller.signal])
  const timeout = setTimeout(() => {
    controller.abort(new Error(`Codex web search timed out after ${REQUEST_TIMEOUT_MS}ms`))
  }, REQUEST_TIMEOUT_MS)
  let onAbort = () => {}
  try {
    signal.throwIfAborted()
    const aborted = new Promise<never>((_, reject) => {
      onAbort = () => reject(signal.reason)
      signal.addEventListener("abort", onAbort, { once: true })
    })
    // Integration resolution has no signal parameter; stop waiting if it outlives the request.
    return await Promise.race([work(signal), aborted])
  } catch (error) {
    if (parent.aborted) throw new Error("Codex web search was cancelled", { cause: error })
    if (controller.signal.aborted) throw controller.signal.reason
    throw error
  } finally {
    clearTimeout(timeout)
    signal.removeEventListener("abort", onAbort)
  }
}

async function requestJson(url: string, auth: CodexAuth, signal: AbortSignal, operation: string, body?: unknown) {
  const headers: Record<string, string> = {
    Accept: "application/json",
    Authorization: `Bearer ${auth.accessToken}`,
    "User-Agent": "codex-cli/0.147.0-alpha.6.5",
  }
  if (auth.accountId) headers["ChatGPT-Account-ID"] = auth.accountId
  if (auth.fedramp) headers["X-OpenAI-Fedramp"] = "true"
  if (body !== undefined) headers["Content-Type"] = "application/json"
  const method = body === undefined ? "GET" : "POST"
  const response = await fetch(url, {
    method,
    headers,
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    signal,
  }).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : "unknown network error"
    throw new Error(`${operation} request failed: ${message}`, { cause: error })
  })
  if (!response.ok) {
    const description = responseError(response, operation)
    // Native errors preserve status/Retry-After for v2's random-provider cooldown and fallback.
    // Deliberately omit authorization headers and request bodies from diagnostic objects.
    const request = HttpClientRequest.make(method)(url)
    const error = new HttpClientError.HttpClientError({
      reason: new HttpClientError.StatusCodeError({
        request,
        response: HttpClientResponse.fromWeb(request, response),
        description,
      }),
    })
    await response.body?.cancel()
    throw error
  }
  return response.json().catch((error: unknown) => {
    throw new Error(`${operation} returned invalid JSON`, { cause: error })
  }) as Promise<unknown>
}

function responseError(response: Response, operation: string) {
  if (response.status === 401) {
    return "OpenCode ChatGPT authentication was rejected or expired; run `opencode auth login` and retry"
  }
  if (response.status === 403 && response.headers.get("cf-mitigated")?.toLowerCase() === "challenge") {
    return `${operation} was blocked by a Cloudflare browser challenge; retry later or from a different network`
  }
  if (response.status === 403) {
    return `${operation} is forbidden for the current ChatGPT account, model, or workspace; verify that this account has Codex access`
  }
  if (response.status === 429) return `${operation} rate limit exceeded; retry later`
  return `${operation} failed with HTTP ${response.status}`
}

async function discoverCodexModel(auth: CodexAuth, signal: AbortSignal) {
  const payload = await requestJson(MODELS_ENDPOINT, auth, signal, "Codex model discovery")
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
      ) return []
      return [{ slug: model.slug, priority: model.priority, visibility: String(model.visibility) }]
    })
    .sort((left, right) => left.priority - right.priority)
  const model = candidates.find((candidate) => candidate.visibility === "list") ?? candidates[0]
  if (!model) throw new Error("Could not determine an account-eligible Codex model for web search; retry later")
  return model.slug
}

function normalizeResponse(payload: unknown, limit: number) {
  if (
    !isRecord(payload) ||
    typeof payload.output !== "string" ||
    (payload.results !== undefined && payload.results !== null && !Array.isArray(payload.results))
  ) throw new Error("Codex web search returned an invalid response")

  const results: { url: string; title: string; content?: string; time: {} }[] = []
  let bytes = 2 // JSON array brackets; include separators in the aggregate budget.
  for (const item of payload.results ?? []) {
    if (!isRecord(item)) continue
    const url = normalizeUrl(item.url)
    if (!url) continue
    const result = { url, title: cleanText(item.title, 300) ?? url, content: cleanText(item.snippet, 1_000), time: {} }
    const size = Buffer.byteLength(JSON.stringify(result), "utf8") + (results.length ? 1 : 0)
    if (bytes + size > MAX_OUTPUT_BYTES) break
    bytes += size
    results.push(result)
    if (results.length >= limit) break
  }
  return results
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function cleanText(value: unknown, maxBytes: number): string | undefined {
  if (typeof value !== "string") return undefined
  const text = value.replace(/\s+/g, " ").trim()
  if (!text) return undefined
  if (Buffer.byteLength(text, "utf8") <= maxBytes) return text
  const segments: string[] = []
  let size = Buffer.byteLength("...", "utf8")
  for (const item of GRAPHEME_SEGMENTER.segment(text)) {
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
