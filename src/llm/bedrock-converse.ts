/**
 * @fileoverview AWS Bedrock Converse API adapter implementing {@link LLMAdapter}.
 *
 * Uses the unified Converse / ConverseStream API from the AWS SDK to provide a
 * single adapter for all non-Anthropic models available through Amazon Bedrock.
 * This covers the following model families:
 *
 *   - **Amazon Nova**   — Nova Micro, Lite, Pro, 2 Lite, Premier
 *   - **Meta Llama**    — Llama 3 / 3.1 / 3.2 / 3.3 / 4 Maverick / 4 Scout
 *   - **Mistral AI**    — Mistral 7B, Large, Small, Pixtral Large, Ministral,
 *                          Devstral, Magistral, Voxtral
 *   - **DeepSeek**      — DeepSeek-R1 (reasoning), V3.1, V3.2
 *   - **NVIDIA**        — Nemotron Nano 9B v2, 12B v2 VL, Nano 3 30B
 *   - **Moonshot AI**   — Kimi K2.5 (vision), Kimi K2 Thinking (reasoning)
 *   - **MiniMax**       — MiniMax M2, M2.1
 *   - **Qwen**          — Qwen3 32B, 235B, Coder 30B/480B/Next, Next 80B,
 *                          VL 235B (vision)
 *   - **Z.AI**          — GLM 4.7, GLM 4.7 Flash
 *   - **Google**        — Gemma 3 4B/12B/27B
 *   - **OpenAI (OSS)**  — gpt-oss-20b, gpt-oss-120b, Safeguard 20B/120B
 *
 * Credential resolution order (standard AWS credential chain):
 *   - Explicit constructor options (`awsAccessKeyId`, `awsSecretAccessKey`, etc.)
 *   - Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
 *     `AWS_SESSION_TOKEN`, `AWS_REGION`)
 *   - IAM role credentials (EC2 instance profile, ECS task role, etc.)
 *
 * EU cross-region inference model IDs (text / chat, `eu.` prefix):
 *   - `eu.amazon.nova-micro-v1:0`   — Amazon Nova Micro (text only, tool use)
 *   - `eu.amazon.nova-lite-v1:0`    — Amazon Nova Lite (multimodal, tool use)
 *   - `eu.amazon.nova-pro-v1:0`     — Amazon Nova Pro (multimodal, tool use)
 *   - `eu.amazon.nova-2-lite-v1:0`  — Amazon Nova 2 Lite (multimodal, tool use)
 *   - `eu.meta.llama3-2-1b-instruct-v1:0` — Meta Llama 3.2 1B (text, no tool use)
 *   - `eu.meta.llama3-2-3b-instruct-v1:0` — Meta Llama 3.2 3B (text, no tool use)
 *   - `eu.mistral.pixtral-large-2502-v1:0` — Mistral Pixtral Large (multimodal, tool use)
 *
 * EU single-region model IDs (no cross-region inference profile — use the
 * region-specific model ID directly and deploy to the appropriate EU region):
 *
 *   NVIDIA Nemotron (eu-south-1, eu-west-1, eu-west-2):
 *   - `nvidia.nemotron-nano-9b-v2`   — text only
 *   - `nvidia.nemotron-nano-12b-v2`  — vision + text
 *   - `nvidia.nemotron-nano-3-30b`   — text only
 *
 *   Moonshot AI Kimi (eu-north-1, eu-west-2):
 *   - `moonshotai.kimi-k2.5`         — vision + text, tool use
 *
 *   MiniMax (eu-central-1, eu-north-1, eu-south-1, eu-west-1, eu-west-2):
 *   - `minimax.minimax-m2`           — text only
 *   - `minimax.minimax-m2.1`         — text only
 *
 *   Qwen (various EU regions):
 *   - `qwen.qwen3-32b-v1:0`               — text only, tool use
 *   - `qwen.qwen3-235b-a22b-2507-v1:0`    — text only, tool use
 *   - `qwen.qwen3-coder-30b-a3b-v1:0`     — text only (code)
 *   - `qwen.qwen3-coder-480b-a35b-v1:0`   — text only (code)
 *   - `qwen.qwen3-coder-next`             — text only (code)
 *   - `qwen.qwen3-next-80b-a3b`           — text only
 *   - `qwen.qwen3-vl-235b-a22b`           — vision + text
 *
 *   DeepSeek (eu-north-1, eu-west-2):
 *   - `deepseek.v3-v1:0`             — text only
 *   - `deepseek.v3.2`                — text only
 *
 *   Z.AI GLM (eu-central-1, eu-north-1, eu-south-1, eu-west-1, eu-west-2):
 *   - `zai.glm-4.7`                  — text only
 *   - `zai.glm-4.7-flash`            — text only
 *
 * Model-specific notes:
 *   - **Reasoning models** (DeepSeek-R1, Kimi K2 Thinking) return
 *     `reasoningContent` blocks in Converse API responses.  These are surfaced
 *     as `TextBlock`s with a `<thinking>…</thinking>` wrapper so downstream
 *     consumers can distinguish reasoning traces from final output.
 *   - **Tool use** is supported natively by Kimi K2.5, Qwen3 (non-Coder),
 *     and most Mistral / Nova / Llama models via the Converse API `toolConfig`
 *     parameter.  Models that lack tool support (e.g. Nemotron, MiniMax,
 *     DeepSeek, Gemma) will ignore `toolConfig` and should not be passed tools.
 *
 * @example
 * ```ts
 * import { BedrockConverseAdapter } from './bedrock-converse.js'
 *
 * const adapter = new BedrockConverseAdapter({ awsRegion: 'eu-central-1' })
 * const response = await adapter.chat(messages, {
 *   model: 'eu.amazon.nova-pro-v1:0',
 *   maxTokens: 2048,
 * })
 * ```
 */

import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime'
import type {
  ContentBlock as BedrockContentBlock,
  ContentBlockDelta as BedrockContentBlockDelta,
  ContentBlockStart as BedrockContentBlockStart,
  Message as BedrockMessage,
  SystemContentBlock as BedrockSystemContentBlock,
  Tool as BedrockTool,
  ToolResultContentBlock as BedrockToolResultContentBlock,
} from '@aws-sdk/client-bedrock-runtime'
import type { DocumentType } from '@smithy/types'

import type {
  ContentBlock,
  ImageBlock,
  LLMAdapter,
  LLMChatOptions,
  LLMMessage,
  LLMResponse,
  LLMStreamOptions,
  LLMToolDef,
  StreamEvent,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
} from '../types.js'

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/** Configuration options for the Bedrock Converse adapter. */
export interface BedrockConverseAdapterOptions {
  /** AWS region. Defaults to `AWS_REGION` env var or `'eu-central-1'`. */
  awsRegion?: string
  /** Explicit AWS access key ID (falls back to default credential chain). */
  awsAccessKeyId?: string
  /** Explicit AWS secret access key (falls back to default credential chain). */
  awsSecretAccessKey?: string
  /** Explicit AWS session token for temporary credentials. */
  awsSessionToken?: string
}

// ---------------------------------------------------------------------------
// Internal helpers – framework → Bedrock
// ---------------------------------------------------------------------------

/**
 * Convert a single framework {@link ContentBlock} into a Bedrock
 * `ContentBlock` suitable for the Converse API `messages` array.
 */
function toBedrockContentBlock(block: ContentBlock): BedrockContentBlock {
  switch (block.type) {
    case 'text':
      return { text: block.text }
    case 'tool_use':
      return {
        toolUse: {
          toolUseId: block.id,
          name: block.name,
          input: block.input as DocumentType,
        },
      }
    case 'tool_result': {
      const resultContent: BedrockToolResultContentBlock[] = [{ text: block.content }]
      return {
        toolResult: {
          toolUseId: block.tool_use_id,
          content: resultContent,
          status: block.is_error ? 'error' : 'success',
        },
      }
    }
    case 'image':
      return {
        image: {
          format: mediaTypeToFormat(block.source.media_type),
          source: {
            bytes: base64ToUint8Array(block.source.data),
          },
        },
      }
    default: {
      const _exhaustive: never = block
      throw new Error(`Unhandled content block type: ${JSON.stringify(_exhaustive)}`)
    }
  }
}

/**
 * Convert framework messages into Bedrock `Message[]` format.
 */
function toBedrockMessages(messages: LLMMessage[]): BedrockMessage[] {
  return messages.map((msg): BedrockMessage => ({
    role: msg.role,
    content: msg.content.map(toBedrockContentBlock),
  }))
}

/**
 * Convert framework {@link LLMToolDef}s into Bedrock `Tool[]`.
 */
function toBedrockTools(tools: readonly LLMToolDef[]): BedrockTool[] {
  return tools.map((t): BedrockTool => ({
    toolSpec: {
      name: t.name,
      description: t.description,
      inputSchema: {
        json: {
          type: 'object',
          ...(t.inputSchema as Record<string, unknown>),
        },
      },
    },
  }))
}

/**
 * Build the system prompt array for the Converse API.
 */
function toBedrockSystem(systemPrompt?: string): BedrockSystemContentBlock[] | undefined {
  if (!systemPrompt) return undefined
  return [{ text: systemPrompt }]
}

// ---------------------------------------------------------------------------
// Internal helpers – Bedrock → framework
// ---------------------------------------------------------------------------

/**
 * Convert a Bedrock `ContentBlock` from the Converse response into a
 * framework {@link ContentBlock}.
 *
 * Reasoning models (DeepSeek-R1, Kimi K2 Thinking) may return
 * `reasoningContent` blocks containing chain-of-thought traces.  These are
 * surfaced as `TextBlock`s wrapped in `<thinking>…</thinking>` tags so that
 * downstream consumers can parse or strip them as needed.
 */
function fromBedrockContentBlock(block: BedrockContentBlock): ContentBlock | null {
  if ('text' in block && block.text !== undefined) {
    const text: TextBlock = { type: 'text', text: block.text }
    return text
  }
  if ('toolUse' in block && block.toolUse !== undefined) {
    const toolUse: ToolUseBlock = {
      type: 'tool_use',
      id: block.toolUse.toolUseId ?? '',
      name: block.toolUse.name ?? '',
      input: (block.toolUse.input as Record<string, unknown>) ?? {},
    }
    return toolUse
  }
  // Reasoning models (e.g. DeepSeek-R1, Kimi K2 Thinking) emit
  // `reasoningContent` blocks with the model's chain-of-thought trace.
  if ('reasoningContent' in block && block.reasoningContent !== undefined) {
    const reasoning = block.reasoningContent as { text?: string }
    if (reasoning.text) {
      const text: TextBlock = {
        type: 'text',
        text: `<thinking>${reasoning.text}</thinking>`,
      }
      return text
    }
  }
  // Graceful degradation for content types we don't yet model.
  return null
}

/**
 * Normalise a Bedrock `StopReason` string into the framework's canonical stop
 * reasons so consumers never need to branch on provider-specific strings.
 */
function normalizeStopReason(reason: string | undefined): string {
  switch (reason) {
    case 'end_turn':
      return 'end_turn'
    case 'tool_use':
      return 'tool_use'
    case 'max_tokens':
      return 'max_tokens'
    case 'stop_sequence':
      return 'stop_sequence'
    case 'content_filtered':
    case 'guardrail_intervened':
      return 'content_filtered'
    default:
      return reason ?? 'end_turn'
  }
}

/**
 * Map MIME type to Bedrock image format string.
 */
function mediaTypeToFormat(
  mediaType: string,
): 'jpeg' | 'png' | 'gif' | 'webp' {
  switch (mediaType) {
    case 'image/jpeg':
      return 'jpeg'
    case 'image/png':
      return 'png'
    case 'image/gif':
      return 'gif'
    case 'image/webp':
      return 'webp'
    default:
      return 'png'
  }
}

/**
 * Decode a base64 string to a Uint8Array.
 */
function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64)
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  return bytes
}

/**
 * Generate a pseudo-unique ID for responses (the Converse API does not return
 * a request-level ID like Anthropic does).
 */
function generateResponseId(): string {
  return `bedrock-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

// ---------------------------------------------------------------------------
// Adapter implementation
// ---------------------------------------------------------------------------

/**
 * LLM adapter backed by the AWS Bedrock Converse API.
 *
 * Supports all models on Amazon Bedrock that expose the Converse / ConverseStream
 * interface, including Amazon Nova, Meta Llama, Mistral, DeepSeek, NVIDIA
 * Nemotron, Moonshot AI Kimi, MiniMax, Qwen, Z.AI GLM, Google Gemma, and
 * OpenAI gpt-oss families.
 *
 * **Reasoning models** (DeepSeek-R1, Kimi K2 Thinking) emit chain-of-thought
 * traces as `<thinking>…</thinking>` wrapped text blocks so that callers can
 * distinguish reasoning from final output.
 *
 * Thread-safe — a single instance may be shared across concurrent agent runs.
 */
export class BedrockConverseAdapter implements LLMAdapter {
  readonly name = 'bedrock-converse'

  readonly #client: BedrockRuntimeClient

  constructor(options?: BedrockConverseAdapterOptions) {
    const region = options?.awsRegion ?? process.env['AWS_REGION'] ?? 'eu-central-1'

    const credentials =
      options?.awsAccessKeyId && options?.awsSecretAccessKey
        ? {
            accessKeyId: options.awsAccessKeyId,
            secretAccessKey: options.awsSecretAccessKey,
            sessionToken: options.awsSessionToken,
          }
        : undefined

    this.#client = new BedrockRuntimeClient({
      region,
      ...(credentials ? { credentials } : {}),
    })
  }

  // -------------------------------------------------------------------------
  // chat()
  // -------------------------------------------------------------------------

  /**
   * Send a synchronous (non-streaming) chat request via the Converse API and
   * return the complete {@link LLMResponse}.
   */
  async chat(messages: LLMMessage[], options: LLMChatOptions): Promise<LLMResponse> {
    const bedrockMessages = toBedrockMessages(messages)

    const command = new ConverseCommand({
      modelId: options.model,
      messages: bedrockMessages,
      system: toBedrockSystem(options.systemPrompt),
      inferenceConfig: {
        maxTokens: options.maxTokens ?? 4096,
        temperature: options.temperature,
      },
      toolConfig: options.tools?.length
        ? { tools: toBedrockTools(options.tools) }
        : undefined,
    })

    const response = await this.#client.send(command, {
      abortSignal: options.abortSignal,
    })

    // Extract content blocks from the response
    const outputMessage =
      response.output && 'message' in response.output
        ? response.output.message
        : undefined

    const content: ContentBlock[] = []
    if (outputMessage?.content) {
      for (const block of outputMessage.content) {
        const converted = fromBedrockContentBlock(block)
        if (converted) content.push(converted)
      }
    }

    return {
      id: generateResponseId(),
      content,
      model: options.model,
      stop_reason: normalizeStopReason(response.stopReason),
      usage: {
        input_tokens: response.usage?.inputTokens ?? 0,
        output_tokens: response.usage?.outputTokens ?? 0,
      },
    }
  }

  // -------------------------------------------------------------------------
  // stream()
  // -------------------------------------------------------------------------

  /**
   * Send a streaming chat request via the ConverseStream API and yield
   * {@link StreamEvent}s as they arrive.
   *
   * Sequence guarantees:
   * - Zero or more `text` events containing incremental deltas
   * - Zero or more `tool_use` events (emitted once per tool call after the
   *   input JSON has been fully assembled)
   * - Exactly one terminal event: `done` or `error`
   */
  async *stream(
    messages: LLMMessage[],
    options: LLMStreamOptions,
  ): AsyncIterable<StreamEvent> {
    const bedrockMessages = toBedrockMessages(messages)

    const command = new ConverseStreamCommand({
      modelId: options.model,
      messages: bedrockMessages,
      system: toBedrockSystem(options.systemPrompt),
      inferenceConfig: {
        maxTokens: options.maxTokens ?? 4096,
        temperature: options.temperature,
      },
      toolConfig: options.tools?.length
        ? { tools: toBedrockTools(options.tools) }
        : undefined,
    })

    const response = await this.#client.send(command, {
      abortSignal: options.abortSignal,
    })

    if (!response.stream) {
      const errorEvent: StreamEvent = {
        type: 'error',
        data: new Error('Bedrock ConverseStream returned no stream'),
      }
      yield errorEvent
      return
    }

    // Accumulate tool-use input JSON and content blocks as they stream in.
    const toolInputBuffers = new Map<
      number,
      { id: string; name: string; json: string }
    >()
    const textBuffers = new Map<number, string>()
    const reasoningBuffers = new Map<number, string>()
    const allContent: ContentBlock[] = []
    let stopReason: string | undefined
    let inputTokens = 0
    let outputTokens = 0

    try {
      for await (const event of response.stream) {
        // --- content_block_start ---
        if ('contentBlockStart' in event && event.contentBlockStart) {
          const idx = event.contentBlockStart.contentBlockIndex ?? 0
          const start: BedrockContentBlockStart | undefined =
            event.contentBlockStart.start
          if (start && 'toolUse' in start && start.toolUse) {
            toolInputBuffers.set(idx, {
              id: start.toolUse.toolUseId ?? '',
              name: start.toolUse.name ?? '',
              json: '',
            })
          }
        }

        // --- content_block_delta ---
        if ('contentBlockDelta' in event && event.contentBlockDelta) {
          const idx = event.contentBlockDelta.contentBlockIndex ?? 0
          const delta: BedrockContentBlockDelta | undefined =
            event.contentBlockDelta.delta

          if (delta) {
            if ('text' in delta && delta.text !== undefined) {
              // Accumulate text for the final response
              const existing = textBuffers.get(idx) ?? ''
              textBuffers.set(idx, existing + delta.text)
              const textEvent: StreamEvent = { type: 'text', data: delta.text }
              yield textEvent
            } else if ('toolUse' in delta && delta.toolUse) {
              const buf = toolInputBuffers.get(idx)
              if (buf !== undefined) {
                buf.json += delta.toolUse.input ?? ''
              }
            } else if ('reasoningContent' in delta && delta.reasoningContent) {
              // Reasoning models (DeepSeek-R1, Kimi K2 Thinking) stream
              // chain-of-thought traces as reasoningContent deltas.
              const rc = delta.reasoningContent as { text?: string }
              if (rc.text) {
                const existing = reasoningBuffers.get(idx) ?? ''
                reasoningBuffers.set(idx, existing + rc.text)
                const textEvent: StreamEvent = { type: 'text', data: rc.text }
                yield textEvent
              }
            }
          }
        }

        // --- content_block_stop ---
        if ('contentBlockStop' in event && event.contentBlockStop) {
          const idx = event.contentBlockStop.contentBlockIndex ?? 0

          // Finalise reasoning blocks (thinking models)
          const reasoningBuf = reasoningBuffers.get(idx)
          if (reasoningBuf !== undefined) {
            allContent.push({
              type: 'text',
              text: `<thinking>${reasoningBuf}</thinking>`,
            })
            reasoningBuffers.delete(idx)
          }

          // Finalise text blocks
          const textBuf = textBuffers.get(idx)
          if (textBuf !== undefined) {
            allContent.push({ type: 'text', text: textBuf })
            textBuffers.delete(idx)
          }

          // Finalise tool-use blocks
          const buf = toolInputBuffers.get(idx)
          if (buf !== undefined) {
            let parsedInput: Record<string, unknown> = {}
            try {
              const parsed: unknown = JSON.parse(buf.json)
              if (
                parsed !== null &&
                typeof parsed === 'object' &&
                !Array.isArray(parsed)
              ) {
                parsedInput = parsed as Record<string, unknown>
              }
            } catch {
              // Malformed JSON — surface as empty object
            }

            const toolUseBlock: ToolUseBlock = {
              type: 'tool_use',
              id: buf.id,
              name: buf.name,
              input: parsedInput,
            }
            allContent.push(toolUseBlock)
            const toolUseEvent: StreamEvent = {
              type: 'tool_use',
              data: toolUseBlock,
            }
            yield toolUseEvent
            toolInputBuffers.delete(idx)
          }
        }

        // --- message_stop ---
        if ('messageStop' in event && event.messageStop) {
          stopReason = event.messageStop.stopReason
        }

        // --- metadata (token usage) ---
        if ('metadata' in event && event.metadata) {
          inputTokens = event.metadata.usage?.inputTokens ?? inputTokens
          outputTokens = event.metadata.usage?.outputTokens ?? outputTokens
        }
      }

      // Build the final response
      const finalResponse: LLMResponse = {
        id: generateResponseId(),
        content: allContent,
        model: options.model,
        stop_reason: normalizeStopReason(stopReason),
        usage: {
          input_tokens: inputTokens,
          output_tokens: outputTokens,
        },
      }

      const doneEvent: StreamEvent = { type: 'done', data: finalResponse }
      yield doneEvent
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      const errorEvent: StreamEvent = { type: 'error', data: error }
      yield errorEvent
    }
  }
}

// Re-export types that consumers of this module commonly need alongside the adapter.
export type {
  ContentBlock,
  ImageBlock,
  LLMAdapter,
  LLMChatOptions,
  LLMMessage,
  LLMResponse,
  LLMStreamOptions,
  LLMToolDef,
  StreamEvent,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
}
