# Open Multi-Agent

> Production-grade multi-agent orchestration framework, extracted from Claude Code's architecture.

[![npm version](https://img.shields.io/npm/v/open-multi-agent)](https://www.npmjs.com/package/open-multi-agent)
[![license](https://img.shields.io/npm/l/open-multi-agent)](./LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue)](https://www.typescriptlang.org/)

## Why Open Multi-Agent?

- **Model-agnostic by design.** One API for Anthropic Claude, OpenAI GPT, or any adapter you write. Switch providers per-agent without changing orchestration code.
- **Production patterns from Claude Code.** The conversation loop engine, tool framework, coordinator mode, and task queue are all adapted from the architecture that powers Claude's own coding agent — not a toy prototype.
- **Zero-boilerplate API.** Describe a goal in plain English. Open Multi-Agent breaks it into tasks, assigns them to the right agents, and resolves dependencies automatically.

---

## Quick Start

```bash
npm install open-multi-agent
```

```typescript
import { OpenMultiAgent } from 'open-multi-agent'

const orchestrator = new OpenMultiAgent({ defaultModel: 'claude-sonnet-4-20250514' })

// One agent, one task
const result = await orchestrator.runAgent(
  {
    name: 'coder',
    model: 'claude-sonnet-4-20250514',
    tools: ['bash', 'file_write'],
  },
  'Write a TypeScript function that reverses a string, save it to /tmp/reverse.ts, and run it.',
)

console.log(result.output)
```

Set `ANTHROPIC_API_KEY` (and optionally `OPENAI_API_KEY`) in your environment before running.

---

## Features

- **Multi-Agent Teams** — Create teams of specialised agents that collaborate toward a shared goal.
- **Automatic Orchestration** — Describe a goal; Open Multi-Agent decomposes it into tasks and assigns them.
- **Tool Framework** — Define custom tools with Zod schemas, or use the 5 built-in tools out of the box.
- **Inter-Agent Communication** — Agents message each other via a typed `MessageBus` and share a `SharedMemory` namespace.
- **Task Pipeline** — Define tasks with `dependsOn` chains; the `TaskQueue` blocks and unblocks them automatically.
- **Model Agnostic** — Works with Anthropic Claude, OpenAI GPT, or any `LLMAdapter` implementation.
- **Parallel Execution** — Independent tasks run concurrently with a configurable `maxConcurrency` cap.
- **Shared Memory** — A namespaced, team-wide key-value store for agent knowledge sharing.
- **Streaming** — Stream incremental text deltas from any agent via `AsyncGenerator<StreamEvent>`.
- **Full Type Safety** — Strict TypeScript with comprehensive TSDoc throughout the codebase.

---

## Usage

### Single Agent

```typescript
import { OpenMultiAgent } from 'open-multi-agent'

const orchestrator = new OpenMultiAgent({ defaultModel: 'claude-sonnet-4-20250514' })

const result = await orchestrator.runAgent(
  {
    name: 'researcher',
    model: 'claude-sonnet-4-20250514',
    systemPrompt: 'You are a rigorous research assistant.',
    tools: ['bash', 'file_read'],
    maxTurns: 8,
  },
  'Summarise the README files of the top 3 TypeScript test runners.',
)

console.log(result.output)
// result.success, result.tokenUsage, result.toolCalls are also available
```

### Streaming Output

Use the `Agent` class directly for streaming (the `OpenMultiAgent` convenience methods return full results):

```typescript
import { Agent, ToolRegistry, ToolExecutor, registerBuiltInTools } from 'open-multi-agent'

const registry = new ToolRegistry()
registerBuiltInTools(registry)
const executor = new ToolExecutor(registry)

const agent = new Agent(
  { name: 'writer', model: 'claude-sonnet-4-20250514', maxTurns: 3 },
  registry,
  executor,
)

for await (const event of agent.stream('Explain monads in two sentences.')) {
  if (event.type === 'text' && typeof event.data === 'string') {
    process.stdout.write(event.data)
  } else if (event.type === 'done') {
    process.stdout.write('\n')
  }
}
```

### Multi-Agent Team

```typescript
import { OpenMultiAgent } from 'open-multi-agent'
import type { AgentConfig } from 'open-multi-agent'

const architect: AgentConfig = {
  name: 'architect',
  model: 'claude-sonnet-4-20250514',
  systemPrompt: 'You design clean API contracts and file structures.',
  tools: ['file_write'],
}

const developer: AgentConfig = {
  name: 'developer',
  model: 'claude-sonnet-4-20250514',
  systemPrompt: 'You implement what the architect designs.',
  tools: ['bash', 'file_read', 'file_write', 'file_edit'],
}

const reviewer: AgentConfig = {
  name: 'reviewer',
  model: 'claude-sonnet-4-20250514',
  systemPrompt: 'You review code for correctness and clarity.',
  tools: ['file_read', 'grep'],
}

const orchestrator = new OpenMultiAgent({
  defaultModel: 'claude-sonnet-4-20250514',
  onProgress: (event) => console.log(event.type, event.agent ?? event.task ?? ''),
})

const team = orchestrator.createTeam('api-team', {
  name: 'api-team',
  agents: [architect, developer, reviewer],
  sharedMemory: true,
})

const result = await orchestrator.runTeam(team, 'Create a REST API for a todo list in /tmp/todo-api/')

console.log(`Success: ${result.success}`)
console.log(`Tokens: ${result.totalTokenUsage.output_tokens} output tokens`)
```

### Task Pipeline

Use `runTasks()` when you want explicit control over the task graph and assignments:

```typescript
const result = await orchestrator.runTasks(team, [
  {
    title: 'Design the data model',
    description: 'Write a TypeScript interface spec to /tmp/spec.md',
    assignee: 'architect',
  },
  {
    title: 'Implement the module',
    description: 'Read /tmp/spec.md and implement the module in /tmp/src/',
    assignee: 'developer',
    dependsOn: ['Design the data model'], // blocked until design completes
  },
  {
    title: 'Write tests',
    description: 'Read the implementation and write Vitest tests.',
    assignee: 'developer',
    dependsOn: ['Implement the module'],
  },
  {
    title: 'Review code',
    description: 'Review /tmp/src/ and produce a structured code review.',
    assignee: 'reviewer',
    dependsOn: ['Implement the module'], // can run in parallel with tests
  },
])
```

The `TaskQueue` resolves dependencies topologically. `Implement the module` is blocked until `Design the data model` finishes. `Write tests` and `Review code` both unblock when the implementation is done — if `maxConcurrency > 1`, they run in parallel.

### Custom Tools

```typescript
import { z } from 'zod'
import { defineTool, Agent, ToolRegistry, ToolExecutor, registerBuiltInTools } from 'open-multi-agent'

const searchTool = defineTool({
  name: 'web_search',
  description: 'Search the web and return the top 5 results.',
  inputSchema: z.object({
    query: z.string().describe('The search query.'),
    maxResults: z.number().optional().describe('Number of results (default 5).'),
  }),
  execute: async ({ query, maxResults = 5 }, context) => {
    // context.agent contains the calling agent's name and model
    const results = await mySearchProvider(query, maxResults)
    return { data: JSON.stringify(results), isError: false }
  },
})

// Build an agent with your custom tool registered in its ToolRegistry
const registry = new ToolRegistry()
registerBuiltInTools(registry)  // add all 5 built-in tools
registry.register(searchTool)   // add your custom tool

const executor = new ToolExecutor(registry)

const agent = new Agent(
  {
    name: 'researcher',
    model: 'claude-sonnet-4-20250514',
    tools: ['web_search'],
  },
  registry,
  executor,
)

const result = await agent.run(
  'Find the three most recent TypeScript releases and summarise their key features.',
)
```

### Multi-Model Teams

```typescript
const claudeAgent: AgentConfig = {
  name: 'strategist',
  model: 'claude-opus-4-5',
  provider: 'anthropic', // explicit, or omit to use the default
  systemPrompt: 'You plan high-level approaches.',
  tools: ['file_write'],
}

const gptAgent: AgentConfig = {
  name: 'implementer',
  model: 'gpt-4o',
  provider: 'openai',
  systemPrompt: 'You implement plans as working code.',
  tools: ['bash', 'file_read', 'file_write'],
}

const team = orchestrator.createTeam('mixed-team', {
  name: 'mixed-team',
  agents: [claudeAgent, gptAgent],
  sharedMemory: true,
})

const result = await orchestrator.runTeam(team, 'Build a CLI tool that converts JSON to CSV.')
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  OpenMultiAgent (Orchestrator)                                  │
│                                                                 │
│  createTeam()  runTeam()  runTasks()  runAgent()  getStatus()   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │  Team               │
            │  - AgentConfig[]    │
            │  - MessageBus       │
            │  - TaskQueue        │
            │  - SharedMemory     │
            │  - EventBus         │
            └──────────┬──────────┘
                       │  creates
         ┌─────────────┴─────────────┐
         │                           │
┌────────▼──────────┐    ┌───────────▼───────────┐
│  AgentPool        │    │  TaskQueue             │
│  - Semaphore      │    │  - dependency graph    │
│  - runParallel()  │    │  - blocked/unblock     │
│  - runAny()       │    │  - event-driven        │
└────────┬──────────┘    └───────────────────────┘
         │ runs
┌────────▼──────────┐
│  Agent            │
│  - run()          │
│  - prompt()       │    ┌──────────────────────┐
│  - stream()       │───►│  LLMAdapter          │
│  - addTool()      │    │  - AnthropicAdapter  │
└────────┬──────────┘    │  - OpenAIAdapter     │
         │               └──────────────────────┘
┌────────▼──────────┐
│  AgentRunner      │
│  - conversation   │    ┌──────────────────────┐
│    loop           │───►│  ToolRegistry        │
│  - tool dispatch  │    │  - defineTool()      │
└───────────────────┘    │  - built-in tools    │
                         └──────────────────────┘
```

---

## API Reference

### `OpenMultiAgent`

```typescript
const orchestrator = new OpenMultiAgent(config: OrchestratorConfig)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `defaultModel` | `string` | — | Model used when an agent doesn't specify one. |
| `defaultProvider` | `'anthropic' \| 'openai'` | `'anthropic'` | Default LLM provider. |
| `maxConcurrency` | `number` | `5` | Max agents running in parallel across the team. |
| `onProgress` | `(event: OrchestratorEvent) => void` | — | Progress callback for lifecycle events. |

**Methods**

```typescript
// Create a named team of agents
orchestrator.createTeam(name: string, config: TeamConfig): Team

// Run a team toward a natural-language goal (OpenMultiAgent decomposes it via coordinator)
await orchestrator.runTeam(team: Team, goal: string): Promise<TeamRunResult>

// Run an explicit list of tasks with optional title-based dependencies
await orchestrator.runTasks(team: Team, tasks: Array<{
  title: string
  description: string
  assignee?: string
  dependsOn?: string[]   // task titles — resolved to IDs automatically
}>): Promise<TeamRunResult>

// Run a single agent on a one-shot prompt
await orchestrator.runAgent(config: AgentConfig, prompt: string): Promise<AgentRunResult>

// Aggregate status across all registered teams
orchestrator.getStatus(): { teams: number; activeAgents: number; completedTasks: number }

// Deregister all teams and reset counters
await orchestrator.shutdown(): Promise<void>
```

### `Team`

```typescript
const team = new Team(config: TeamConfig)
```

```typescript
// Agent roster
team.getAgents(): AgentConfig[]
team.getAgent(name: string): AgentConfig | undefined

// Messaging
team.sendMessage(from: string, to: string, content: string): void
team.broadcast(from: string, content: string): void
team.getMessages(agentName: string): Message[]

// Task management
team.addTask(task: Omit<Task, 'id' | 'createdAt' | 'updatedAt'>): Task
team.getTasks(): Task[]
team.getNextTask(agentName: string): Task | undefined
team.updateTask(taskId: string, update: Partial<Task>): Task

// Shared memory
team.getSharedMemory(): MemoryStore | undefined

// Events
team.on(event: string, handler: (data: unknown) => void): () => void
team.emit(event: string, data: unknown): void
```

**Team events**

| Event | Fires when |
|-------|-----------|
| `task:ready` | A task becomes runnable (deps satisfied). |
| `task:complete` | A task finishes successfully. |
| `task:failed` | A task fails. |
| `all:complete` | Every task in the queue has terminated. |
| `message` | A point-to-point message is sent. |
| `broadcast` | A broadcast message is sent. |

### `Agent`

```typescript
// One-shot run (no conversation history retained)
await agent.run(prompt: string): Promise<AgentRunResult>

// Multi-turn conversation (history is retained across calls)
await agent.prompt(message: string): Promise<AgentRunResult>

// Streaming (no history retained)
agent.stream(prompt: string): AsyncGenerator<StreamEvent>

// Dynamic tool management
agent.addTool(tool: ToolDefinition): void
agent.removeTool(name: string): void
agent.getTools(): string[]

// State
agent.getState(): AgentState    // { status, messages, tokenUsage }
agent.getHistory(): LLMMessage[]
agent.reset(): void
```

### `defineTool()`

```typescript
import { z } from 'zod'
import { defineTool } from 'open-multi-agent'

const myTool = defineTool({
  name: 'my_tool',
  description: 'What this tool does.',
  inputSchema: z.object({
    param: z.string().describe('Description of param.'),
  }),
  execute: async (input, context) => {
    // context.agent: { name, role, model }
    // context.team: TeamInfo (when running inside a team)
    // context.abortSignal: AbortSignal
    return { data: 'result string', isError: false }
  },
})
```

### `TaskQueue`

```typescript
const queue = new TaskQueue()

// Add tasks
queue.add(task: Task): void
queue.addBatch(tasks: Task[]): void

// Lifecycle
queue.complete(taskId: string, result?: string): Task
queue.fail(taskId: string, error: string): Task
queue.update(taskId, { status?, result?, assignee? }): Task

// Queries
queue.next(assignee?: string): Task | undefined
queue.nextAvailable(): Task | undefined
queue.list(): Task[]
queue.getByStatus(status: TaskStatus): Task[]
queue.isComplete(): boolean
queue.getProgress(): { total, completed, failed, inProgress, pending, blocked }

// Events
queue.on('task:ready', (task) => { ... })
queue.on('task:complete', (task) => { ... })
queue.on('task:failed', (task) => { ... })
queue.on('all:complete', () => { ... })
```

### `SharedMemory`

```typescript
const mem = new SharedMemory()

// All writes are namespaced: stored as "<agentName>/<key>"
await mem.write(agentName: string, key: string, value: string): Promise<void>
await mem.read(fullyQualifiedKey: string): Promise<MemoryEntry | null>
await mem.listAll(): Promise<MemoryEntry[]>
await mem.listByAgent(agentName: string): Promise<MemoryEntry[]>
await mem.getSummary(): Promise<string>  // human-readable digest for context injection
```

---

## Built-in Tools

All five tools are registered automatically when you use the `OpenMultiAgent` class. Reference them by name in `AgentConfig.tools`.

| Tool name | Description |
|-----------|-------------|
| `bash` | Execute any shell command. Returns stdout + stderr. Supports `timeout` and `cwd`. |
| `file_read` | Read the contents of a file at an absolute path. |
| `file_write` | Write or create a file at an absolute path. |
| `file_edit` | Edit a file by replacing an exact string with a new one. |
| `grep` | Search file contents with a regular expression. Returns matching lines. |

---

## Comparison

| Feature | Open Multi-Agent | LangChain | CrewAI | AutoGen |
|---------|-----------------|-----------|--------|---------|
| Model Agnostic | Yes | Yes | No | Yes |
| In-Process (no server) | Yes | Yes | Yes | Yes |
| Production-Grade File Tools | Yes | No | No | No |
| Task Dependency Graph | Yes | No | No | No |
| Inter-Agent Messaging | Yes | No | Yes | Yes |
| Shared Team Memory | Yes | No | No | No |
| Streaming Support | Yes | Partial | No | No |
| Zero Config Start | Yes | No | No | No |
| TypeScript-First | Yes | Partial | No | Partial |

---

## Inspired By

This framework extracts and generalises the multi-agent orchestration patterns from Claude Code's production architecture — the same system that powers Claude's coding capabilities in Claude Code.

Key patterns adapted:

| Claude Code internal | open-multi-agent equivalent | What it does |
|---------------------|----------------------------|--------------|
| `QueryEngine` conversation loop | `AgentRunner` | Drives the model → tool → model turn loop |
| `buildTool()` | `defineTool()` | Typed tool definition with Zod validation |
| Coordinator mode | `OpenMultiAgent` orchestrator | Decomposes goals, assigns tasks, manages concurrency |
| Team / sub-agent system | `Team` + `MessageBus` | Inter-agent communication and shared state |
| Task tracking | `TaskQueue` with dependency resolution | Topological task scheduling |

The architecture is intentionally kept minimal — no heavyweight external dependencies beyond the provider SDKs (`@anthropic-ai/sdk`, `openai`) and `zod`. The entire framework is importable from a single entry point.

---

## License

MIT
