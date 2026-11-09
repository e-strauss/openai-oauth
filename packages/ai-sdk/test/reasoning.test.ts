import type { LanguageModelV3CallOptions } from "@ai-sdk/provider"
import { describe, expect, test, vi } from "vitest"
import { createOpenAIOAuth } from "../src/index.js"

describe("reasoning model detection", () => {
	test.each([
		{ modelId: "gpt-6-luna", forceReasoning: undefined, reasoning: true },
		{ modelId: "gpt-6.1-sol", forceReasoning: undefined, reasoning: true },
		{ modelId: "gpt-5.6-sol", forceReasoning: undefined, reasoning: true },
		{ modelId: "gpt-4o", forceReasoning: undefined, reasoning: false },
		{ modelId: "gpt-60-custom", forceReasoning: undefined, reasoning: false },
		{ modelId: "gpt-6-luna", forceReasoning: false, reasoning: false },
		{ modelId: "custom-model", forceReasoning: true, reasoning: true },
	])("$modelId with forceReasoning=$forceReasoning", async (scenario) => {
		const bodies: Array<Record<string, unknown>> = []
		const fetch = vi.fn(
			async (_input: RequestInfo | URL, init?: RequestInit) => {
				bodies.push(JSON.parse(String(init?.body)))
				return new Response(
					`data: ${JSON.stringify({
						type: "response.completed",
						response: {
							id: "offline",
							model: scenario.modelId,
							output: [],
							usage: { input_tokens: 0, output_tokens: 0 },
						},
					})}\n\n`,
					{ headers: { "Content-Type": "text/event-stream" } },
				)
			},
		)
		// Capture the SDK conversion before core can fill in catalog defaults.
		const provider = createOpenAIOAuth({
			kind: "openai-compatible",
			baseURL: "https://offline.invalid/v1",
			fetch,
			request: async () => {
				throw new Error("Unexpected transport request")
			},
		})
		const model = provider(scenario.modelId)
		const options: LanguageModelV3CallOptions = {
			prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
			providerOptions: {
				openai: {
					reasoningEffort: "low",
					parallelToolCalls: false,
					...(scenario.forceReasoning === undefined
						? {}
						: { forceReasoning: scenario.forceReasoning }),
				},
			},
		}
		const originalOptions = structuredClone(options)
		const generated = await model.doGenerate(options)
		const streamed = await model.doStream(options)
		const warnings = [...generated.warnings]
		const reader = streamed.stream.getReader()
		try {
			while (true) {
				const { done, value } = await reader.read()
				if (done) break
				if (value.type === "stream-start") {
					warnings.push(...value.warnings)
				}
			}
		} finally {
			reader.releaseLock()
		}

		expect(bodies).toHaveLength(2)
		for (const body of bodies) {
			expect(body.parallel_tool_calls).toBe(false)
			if (scenario.reasoning) {
				expect(body.reasoning).toEqual({ effort: "low" })
			} else {
				expect(body.reasoning).toBeUndefined()
			}
		}
		if (scenario.reasoning) {
			expect(warnings).toEqual([])
		} else {
			expect(warnings).toContainEqual(
				expect.objectContaining({
					type: "unsupported",
					feature: "reasoningEffort",
				}),
			)
		}
		expect(options).toEqual(originalOptions)
	})
})
