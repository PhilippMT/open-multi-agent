/**
 * @fileoverview Amazon Bedrock Anthropic Claude adapter implementing {@link LLMAdapter}.
 *
 * This adapter is functionally identical to the standard {@link AnthropicAdapter}
 * but authenticates via AWS IAM / STS credentials instead of an Anthropic API key.
 * It uses the `@anthropic-ai/bedrock-sdk` package which wraps the standard
 * Anthropic SDK with Bedrock-native authentication.
 *
 * Credential resolution order (all handled by the SDK / AWS credential chain):
 *   - Explicit constructor options (`awsAccessKeyId`, `awsSecretAccessKey`, etc.)
 *   - Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
 *     `AWS_SESSION_TOKEN`, `AWS_REGION`)
 *
 * EU cross-region inference model IDs (format varies per model release):
 *   - `eu.anthropic.claude-haiku-4-5-20251001-v1:0`
 *   - `eu.anthropic.claude-sonnet-4-5-20250929-v1:0`
 *   - `eu.anthropic.claude-sonnet-4-6`
 *   - `eu.anthropic.claude-opus-4-5-20251101-v1:0`
 *   - `eu.anthropic.claude-opus-4-6-v1`
 *
 * @example
 * ```ts
 * import { BedrockAnthropicAdapter } from './bedrock-anthropic.js'
 *
 * const adapter = new BedrockAnthropicAdapter({ awsRegion: 'eu-central-1' })
 * const response = await adapter.chat(messages, {
 *   model: 'eu.anthropic.claude-sonnet-4-6',
 *   maxTokens: 1024,
 * })
 * ```
 */

import AnthropicBedrock from '@anthropic-ai/bedrock-sdk'
import type {
  ContentBlock as AnthropicContentBlock,
  ContentBlockParam,
  ImageBlockParam,
  MessageParam,
  TextBlockParam,
  ToolResultBlockParam,
  ToolUseBlockParam,
  Tool as AnthropicTool,
} from '@anthropic-ai/sdk/resources/messages/messages.js'

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

export interface BedrockAnthropicAdapterOptions {
  awsRegion?: string
  awsAccessKeyId?: string
  awsSecretAccessKey?: string
  awsSessionToken?: string
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function toAnthropicContentBlockParam(block: ContentBlock): ContentBlockParam {
  switch (block.type) {
    case 'text': {
      const param: TextBlockParam = { type: 'text', text: block.text }
      return param
    }
    case 'tool_use': {
      const param: ToolUseBlockParam = {
        type: 'tool_use',
        id: block.id,
        name: block.name,
        input: block.input,
      }
      return param
    }
    case 'tool_result': {
      const param: ToolResultBlockParam = {
        type: 'tool_result',
        tool_use_id: block.tool_use_id,
        content: block.content,
        is_error: block.is_error,
      }
      return param
    }
    case 'image': {
      const param: ImageBlockParam = {
        type: 'image',
        source: {
          type: 'base64',
          media_type: block.source.media_type as
            | 'image/jpeg'
            | 'image/png'
            | 'image/gif'
            | 'image/webp',
          data: block.source.data,
        },
      }
      return param
    }
    default: {
      const _exhaustive: never = block
      throw new Error(`Unhandled content block type: ${JSON.stringify(_exhaustive)}`)
    }
  }
}

function toAnthropicMessages(messages: LLMMessage[]): MessageParam[] {
  return messages.map((msg): MessageParam => ({
    role: msg.role,
    content: msg.content.map(toAnthropicContentBlockParam),
  }))
}

function toAnthropicTools(tools: readonly LLMToolDef[]): AnthropicTool[] {
  return tools.map((t): AnthropicTool => ({
    name: t.name,
    description: t.description,
    input_schema: {
      type: 'object',
      ...(t.inputSchema as Record<string, unknown>),
    },
  }))
}

function fromAnthropicContentBlock(
  block: AnthropicContentBlock,
): ContentBlock {
  switch (block.type) {
    case 'text': {
      const text: TextBlock = { type: 'text', text: block.text }
      return text
    }
    case 'tool_use': {
      const toolUse: ToolUseBlock = {
        type: 'tool_use',
        id: block.id,
        name: block.name,
        input: block.input as Record<string, unknown>,
      }
      return toolUse
    }
    default: {
      const fallback: TextBlock = {
        type: 'text',
        text: `[unsupported block type: ${(block as { type: string }).type}]`,
      }
      return fallback
    }
  }
}

// ---------------------------------------------------------------------------
// Adapter implementation
// ---------------------------------------------------------------------------

export class BedrockAnthropicAdapter implements LLMAdapter {
  readonly name = 'bedrock-anthropic'

  readonly #client: AnthropicBedrock

  constructor(options?: BedrockAnthropicAdapterOptions) {
    const awsAccessKey = options?.awsAccessKeyId
    const awsSecretKey = options?.awsSecretAccessKey
    const awsRegion = options?.awsRegion ?? process.env['AWS_REGION'] ?? 'eu-central-1'
    const awsSessionToken = options?.awsSessionToken

    if (awsAccessKey && awsSecretKey) {
      this.#client = new AnthropicBedrock({
        awsRegion,
        awsAccessKey,
        awsSecretKey,
        awsSessionToken,
      })
    } else {
      this.#client = new AnthropicBedrock({
        awsRegion,
      })
    }
  }

  async chat(messages: LLMMessage[], options: LLMChatOptions): Promise<LLMResponse> {
    const anthropicMessages = toAnthropicMessages(messages)

    const response = await this.#client.messages.create(
      {
        model: options.model,
        max_tokens: options.maxTokens ?? 4096,
        messages: anthropicMessages,
        system: options.systemPrompt,
        tools: options.tools ? toAnthropicTools(options.tools) : undefined,
        temperature: options.temperature,
      },
      {
        signal: options.abortSignal,
      },
    )

    const content = response.content.map(fromAnthropicContentBlock)

    return {
      id: response.id,
      content,
      model: response.model,
      stop_reason: response.stop_reason ?? 'end_turn',
      usage: {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
      },
    }
  }

  async *stream(
    messages: LLMMessage[],
    options: LLMStreamOptions,
  ): AsyncIterable<StreamEvent> {
    const anthropicMessages = toAnthropicMessages(messages)

    const stream = this.#client.messages.stream(
      {
        model: options.model,
        max_tokens: options.maxTokens ?? 4096,
        messages: anthropicMessages,
        system: options.systemPrompt,
        tools: options.tools ? toAnthropicTools(options.tools) : undefined,
        temperature: options.temperature,
      },
      {
        signal: options.abortSignal,
      },
    )

    const toolInputBuffers = new Map<number, { id: string; name: string; json: string }>()

    try {
      for await (const event of stream) {
        switch (event.type) {
          case 'content_block_start': {
            const block = event.content_block
            if (block.type === 'tool_use') {
              toolInputBuffers.set(event.index, {
                id: block.id,
                name: block.name,
                json: '',
              })
            }
            break
          }

          case 'content_block_delta': {
            const delta = event.delta

            if (delta.type === 'text_delta') {
              const textEvent: StreamEvent = { type: 'text', data: delta.text }
              yield textEvent
            } else if (delta.type === 'input_json_delta') {
              const buf = toolInputBuffers.get(event.index)
              if (buf !== undefined) {
                buf.json += delta.partial_json
              }
            }
            break
          }

          case 'content_block_stop': {
            const buf = toolInputBuffers.get(event.index)
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
              const toolUseEvent: StreamEvent = { type: 'tool_use', data: toolUseBlock }
              yield toolUseEvent
              toolInputBuffers.delete(event.index)
            }
            break
          }

          default:
            break
        }
      }

      const finalMessage = await stream.finalMessage()
      const content = finalMessage.content.map(fromAnthropicContentBlock)

      const finalResponse: LLMResponse = {
        id: finalMessage.id,
        content,
        model: finalMessage.model,
        stop_reason: finalMessage.stop_reason ?? 'end_turn',
        usage: {
          input_tokens: finalMessage.usage.input_tokens,
          output_tokens: finalMessage.usage.output_tokens,
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
