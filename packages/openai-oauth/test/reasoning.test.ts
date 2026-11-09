import { promises as fs } from "node:fs"
import os from "node:os"
import path from "node:path"
import { describe, expect, test } from "vitest"
import { createOpenAIOAuthFetchHandler } from "../src/index.js"

const completedResponse = (model: string): Response => {
	const message = {
		type: "message",
		id: "msg_1",
		role: "assistant",
		status: "completed",
		content: [{ type: "output_text", text: "hello", annotations: [] }],
	}
	const response = {
		id: "resp_1",
		model,
		created_at: 1735689600,
		status: "completed",
		output: [message],
		usage: { input_tokens: 1, output_tokens: 1 },
	}
	const events = [
		{ type: "response.created", response: { ...response, output: [] } },
		{
			type: "response.output_item.added",
			output_index: 0,
			item: { ...message, content: [] },
		},
		{
			type: "response.output_text.delta",
			item_id: message.id,
			output_index: 0,
			content_index: 0,
			delta: "hello",
		},
		{ type: "response.output_item.done", output_index: 0, item: message },
		{ type: "response.completed", response },
	]
	return new Response(
		events
			.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
			.join(""),
		{ headers: { "Content-Type": "text/event-stream" } },
	)
}

const cases = ["gpt-6-luna", "gpt-6.1-sol"].flatMap((model) =>
	[false, true].flatMap((stream) =>
		["low", "none", undefined].map((effort) => ({ model, stream, effort })),
	),
)

describe("Chat Completions reasoning forwarding", () => {
	test.each(cases)(
		"$model stream=$stream reasoning_effort=$effort",
		async ({ model, stream, effort }) => {
			const directory = await fs.mkdtemp(
				path.join(os.tmpdir(), "openai-oauth-reasoning-"),
			)
			try {
				const authFilePath = path.join(directory, "auth.json")
				await fs.writeFile(
					authFilePath,
					JSON.stringify({
						tokens: { access_token: "offline-token", account_id: "offline" },
					}),
				)
				const bodies: Array<Record<string, unknown>> = []
				const handler = createOpenAIOAuthFetchHandler({
					authFilePath,
					ensureFresh: false,
					codexVersion: "0.144.1",
					fetch: async (input, init) => {
						const url = String(input)
						if (url.includes("/backend-api/codex/models?")) {
							return Response.json({
								models: [
									{
										slug: model,
										visibility: "list",
										default_reasoning_level: "high",
									},
								],
							})
						}
						if (url.endsWith("/backend-api/codex/responses")) {
							bodies.push(JSON.parse(String(init?.body)))
							return completedResponse(model)
						}
						throw new Error(`Unexpected offline request: ${url}`)
					},
				})
				const response = await handler(
					new Request("http://localhost/v1/chat/completions", {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({
							model,
							stream,
							reasoning_effort: effort,
							messages: [{ role: "user", content: "hello" }],
						}),
					}),
				)
				const result = await response.text()
				expect(response.status, result).toBe(200)
				if (stream) {
					expect(result).toContain('"content":"hello"')
					expect(result).toContain("data: [DONE]")
				} else {
					expect(JSON.parse(result)).toMatchObject({
						choices: [{ message: { content: "hello" }, finish_reason: "stop" }],
					})
				}
				expect(bodies).toHaveLength(1)
				expect(bodies[0]).toMatchObject({
					model,
					reasoning: { effort: effort ?? "high" },
				})
			} finally {
				await fs.rm(directory, { recursive: true, force: true })
			}
		},
	)
})
