import assert from "node:assert/strict"
import { test } from "node:test"
import type { TestContext } from "node:test"
import { setImmediate } from "node:timers/promises"
import { Credential, Integration, WebSearch } from "@opencode/plugin"
import type { Plugin } from "@opencode/plugin"
import type { WebSearchDefinition, WebSearchEditor } from "@opencode/plugin/promise/websearch"
import type { IntegrationDomain } from "@opencode/plugin/promise/integration"
import { Schema } from "effect"
import { HttpClientError } from "effect/unstable/http"
import plugin from "opencode-codex-websearch"

function oauth(updates: Partial<Credential.OAuth> = {}) {
  return Credential.OAuth.make({
    type: "oauth",
    methodID: Integration.MethodID.make("chatgpt-browser"),
    access: "synthetic-access",
    refresh: "synthetic-refresh",
    expires: Date.now() + 3_600_000,
    metadata: { accountID: "account-a" },
    ...updates,
  })
}

async function fixture(t: TestContext, options: Record<string, unknown> = {}) {
  let value: Credential.Value | undefined = oauth()
  let active: Awaited<ReturnType<IntegrationDomain["connection"]["active"]>> = {
    type: "credential", id: Credential.ID.make("cred_test"), label: "ChatGPT",
  }
  const calls: string[] = []
  let provider: WebSearchDefinition | undefined
  let replay: ((editor: WebSearchEditor) => void) | undefined
  const editor: WebSearchEditor = {
    add(definition) { provider = definition },
    default: { get: () => "existing-provider", set: () => assert.fail("Must preserve provider selection") },
  }
  const connection: IntegrationDomain["connection"] = {
    async active(id) { calls.push(`active:${id}`); return active },
    async resolve(selected) { assert.equal(selected, active); calls.push("resolve"); return value },
  }
  const context = {
    options,
    integration: { connection },
    websearch: {
      async transform(callback: (editor: WebSearchEditor) => void) {
        replay = callback
        assert.equal(callback(editor), undefined, "Transform must be synchronous")
        return { async dispose() { provider = undefined } }
      },
    },
  } as unknown as Plugin.Context
  const cleanup = await plugin.setup(context)
  assert.equal(typeof cleanup, "function")
  t.after(() => cleanup?.())
  assert.ok(provider)
  const definition = provider
  return {
    calls, connection, definition,
    cleanup: () => cleanup?.(),
    replay: () => replay?.(editor),
    setCredential(next: Credential.Value | undefined) { value = next },
    setActive(next: typeof active) { active = next },
    search: (query = "OpenCode", signal = new AbortController().signal) => definition.execute({ query }, { signal }),
  }
}

const models = { models: [
  { slug: "hidden", visibility: "hide", priority: 0 },
  { slug: "gpt-eligible", visibility: "list", priority: 1 },
] }
const search = { output: "Ignored upstream prose", results: [
  { title: " OpenCode\n docs ", url: "https://opencode.ai/v2/docs", snippet: " Current\n documentation. " },
] }

function http(t: TestContext, respond?: (url: string, init: RequestInit) => Response | Promise<Response>) {
  const requests: { url: string; init: RequestInit }[] = []
  t.mock.method(globalThis, "fetch", async (url: string, init: RequestInit) => {
    requests.push({ url, init })
    return respond ? respond(url, init) : Response.json(url.includes("/models?") ? models : search)
  })
  return requests
}

test("registers a replayable native provider and returns schema-valid results", async (t) => {
  const requests = http(t)
  const f = await fixture(t, { max_results: 2, recency: 30, domains: [" github.com "] })
  assert.equal(plugin.id, "opencode-codex-websearch")
  assert.equal(f.definition.id, "codex")
  assert.equal(f.definition.name, "ChatGPT Codex")
  f.replay()
  assert.deepEqual(f.calls, [])
  assert.equal(requests.length, 0)
  const results = await f.search("  OpenCode  ")
  assert.deepEqual(results, [{ url: "https://opencode.ai/v2/docs", title: "OpenCode docs", content: "Current documentation.", time: {} }])
  assert.ok(Schema.is(Schema.Array(WebSearch.Result))(results))
  assert.deepEqual(f.calls, ["active:openai", "resolve"])
  const request = requests[1]!
  const body = JSON.parse(String(request.init.body))
  assert.match(body.id, /^search_session_[a-f0-9]{16}$/)
  assert.equal(body.model, "gpt-eligible")
  assert.deepEqual(body.commands.search_query, [{ q: "OpenCode", recency: 30, domains: ["github.com"] }])
  assert.equal(request.url, "https://chatgpt.com/backend-api/codex/alpha/search")
  const headers = new Headers(request.init.headers)
  assert.equal(headers.get("Authorization"), "Bearer synthetic-access")
  assert.equal(headers.get("ChatGPT-Account-ID"), "account-a")
})

test("resolves fresh tokens every call and rediscovers models on account/credential changes", async (t) => {
  const requests = http(t)
  const f = await fixture(t)
  await f.search()
  f.setCredential(oauth({ access: "refreshed-access" }))
  await f.search()
  assert.equal(requests.length, 3)
  assert.equal(new Headers(requests[2]!.init.headers).get("Authorization"), "Bearer refreshed-access")
  f.setCredential(oauth({ metadata: { accountID: "account-b" } }))
  await f.search()
  assert.equal(requests.length, 5)
  f.setActive({ type: "credential", id: Credential.ID.make("cred_other"), label: "Other" })
  await f.search()
  assert.equal(requests.length, 7)
  const other = await fixture(t)
  await other.search()
  assert.equal(requests.length, 9, "Model cache must be instance-scoped")
})

test("uses metadata accountID before JWT fallback and supports headless/FedRAMP claims", async (t) => {
  const requests = http(t)
  const token = `header.${Buffer.from(JSON.stringify({ "https://api.openai.com/auth": {
    chatgpt_account_id: "claim-account", chatgpt_account_is_fedramp: true,
  } })).toString("base64url")}.signature`
  const f = await fixture(t)
  f.setCredential(oauth({ access: token, methodID: Integration.MethodID.make("chatgpt-headless") }))
  await f.search()
  assert.equal(new Headers(requests[0]!.init.headers).get("ChatGPT-Account-ID"), "account-a")
  f.setCredential(oauth({ access: token, metadata: undefined }))
  await f.search()
  const headers = new Headers(requests[2]!.init.headers)
  assert.equal(headers.get("ChatGPT-Account-ID"), "claim-account")
  assert.equal(headers.get("X-OpenAI-Fedramp"), "true")
})

test("missing, API-key, unsupported OAuth, and expired credentials fail before HTTP", async (t) => {
  const requests = http(t)
  const f = await fixture(t)
  for (const credential of [undefined, Credential.Key.make({ type: "key", key: "api-key" }),
    oauth({ methodID: Integration.MethodID.make("other-oauth") }), oauth({ access: "" })]) {
    f.setCredential(credential)
    await assert.rejects(f.search(), /authentication is unavailable/)
  }
  f.setCredential(oauth({ expires: Date.now() - 1 }))
  await assert.rejects(f.search(), /authentication has expired/)
  f.setActive(undefined)
  await assert.rejects(f.search(), /authentication is unavailable/)
  assert.equal(requests.length, 0)
})

test("credential refresh failures propagate without making search requests", async (t) => {
  const requests = http(t)
  const f = await fixture(t)
  t.mock.method(f.connection, "resolve", async () => { throw new Error("OAuth refresh failed") })
  await assert.rejects(f.search(), /OAuth refresh failed/)
  assert.equal(requests.length, 0)
})

test("validates options at setup and queries before resolving credentials", async (t) => {
  http(t)
  for (const options of [
    { max_results: 0 }, { max_results: 21 }, { max_results: 1.5 }, { max_results: "8" },
    { recency: 0 }, { recency: 3651 }, { recency: null }, { domains: "github.com" },
    { domains: [""] }, { domains: [5] }, { domains: ["x".repeat(254)] },
    { domains: Array(21).fill("github.com") }, { typo: true },
  ]) await assert.rejects(fixture(t, options), /Codex search/)
  const f = await fixture(t)
  for (const query of ["", " \n ", "x".repeat(501)]) await assert.rejects(f.search(query), /query must contain/)
  assert.deepEqual(f.calls, [])
})

test("defaults to eight results, omits empty filters, and sanitizes output", async (t) => {
  const requests = http(t, (url) => Response.json(url.includes("/models?") ? models : {
    output: "Not returned",
    results: [null, { url: "javascript:alert(1)" }, { url: "ftp://example.com" },
      ...Array.from({ length: 12 }, (_, n) => ({ url: `https://example.com/${n}`, title: "😀".repeat(300), snippet: "👩‍💻".repeat(200) }))],
  }))
  const f = await fixture(t, { domains: [] })
  const results = await f.search()
  assert.equal(results.length, 8)
  for (const item of results) {
    assert.ok(Buffer.byteLength(item.title!) <= 300)
    assert.ok(Buffer.byteLength(item.content!) <= 1000)
    assert.match(item.content!, /^(👩‍💻)+\.\.\.$/u)
  }
  assert.deepEqual(JSON.parse(String(requests[1]!.init.body)).commands.search_query, [{ q: "OpenCode" }])
})

test("bounds aggregate results and preserves empty results", async (t) => {
  http(t, (url) => Response.json(url.includes("/models?") ? models : {
    output: "", results: Array.from({ length: 20 }, (_, n) => ({
      url: `https://example.com/${n}/${"a".repeat(1900)}`, title: "t".repeat(300), snippet: "s".repeat(1000),
    })),
  }))
  const f = await fixture(t, { max_results: 20 })
  assert.ok(Buffer.byteLength(JSON.stringify(await f.search())) <= 20_000)
  http(t, () => Response.json({ output: "", results: null }))
  assert.deepEqual(await f.search(), [])
})

test("rejects malformed discovery/search responses and invalid JSON", async (t) => {
  const f = await fixture(t)
  for (const body of [{ models: [] }, { models: [{ slug: "bad", priority: "1", visibility: "list" }] }, {}]) {
    http(t, () => Response.json(body))
    await assert.rejects(f.search(), /model|response/)
  }
  http(t, () => new Response("{"))
  await assert.rejects(f.search(), /invalid JSON/)
  http(t, (url) => Response.json(url.includes("/models?") ? models : { output: "", results: {} }))
  await assert.rejects(f.search(), /invalid response/)
  http(t, () => new Response("{"))
  await assert.rejects(f.search(), /web search returned invalid JSON/)
})

test("preserves native HTTP statuses, retry headers, and useful errors without credential leakage", async (t) => {
  const f = await fixture(t)
  http(t)
  await f.search()
  for (const [status, message] of [[401, /authentication was rejected/], [403, /forbidden/], [429, /rate limit/], [500, /HTTP 500/]] as const) {
    http(t, () => new Response("", { status, headers: { "Retry-After": "90" } }))
    await assert.rejects(f.search(), (error: unknown) => {
      assert.ok(HttpClientError.isHttpClientError(error))
      assert.equal(error.response?.status, status)
      assert.equal(error.response?.headers["retry-after"], "90")
      assert.match(error.message, message)
      assert.ok(!JSON.stringify(error).includes("synthetic-access"))
      return true
    })
  }
  http(t, () => new Response("", { status: 403, headers: { "cf-mitigated": "challenge" } }))
  await assert.rejects(f.search(), /Cloudflare browser challenge/)
})

test("reports network failures", async (t) => {
  http(t, () => { throw new Error("offline") })
  const f = await fixture(t)
  await assert.rejects(f.search(), /model discovery request failed: offline/)
})

test("pre-aborted searches do not resolve credentials", async (t) => {
  const requests = http(t)
  const f = await fixture(t)
  await assert.rejects(f.search("query", AbortSignal.abort()), /cancelled/)
  assert.deepEqual(f.calls, [])
  assert.equal(requests.length, 0)
})

test("cancellation and unload abort pending HTTP requests", async (t) => {
  const signals: AbortSignal[] = []
  http(t, async (_, init) => {
    signals.push(init.signal!)
    return new Promise<Response>((_, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason), { once: true }))
  })
  const f = await fixture(t)
  const controller = new AbortController()
  const pending = f.search("query", controller.signal)
  await setImmediate()
  controller.abort()
  await assert.rejects(pending, /cancelled/)
  assert.equal(signals[0]!.aborted, true)
  const unloading = f.search()
  await setImmediate()
  await f.cleanup()
  await assert.rejects(unloading, /cancelled/)
  assert.equal(signals[1]!.aborted, true)
})

test("deadline aborts HTTP and also stops waiting on credential resolution", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] })
  const requests = http(t, async (_, init) => new Promise<Response>((_, reject) => {
    init.signal!.addEventListener("abort", () => reject(init.signal!.reason), { once: true })
  }))
  const f = await fixture(t)
  const pending = f.search()
  await setImmediate()
  t.mock.timers.tick(15_000)
  await assert.rejects(pending, /timed out after 15000ms/)
  assert.equal(requests[0]!.init.signal!.aborted, true)

  let resolve!: (value: Credential.Value) => void
  t.mock.method(f.connection, "resolve", () => new Promise<Credential.Value>((done) => { resolve = done }))
  const authPending = f.search()
  await setImmediate()
  t.mock.timers.tick(15_000)
  await assert.rejects(authPending, /timed out/)
  resolve(oauth())
  await setImmediate()
  assert.equal(requests.length, 1, "Late credentials must not start HTTP after timeout")
})
