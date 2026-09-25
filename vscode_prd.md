Product Requirements Document (PRD)
Product: ck for VS Code / Cursor
Version: v1.0
Author: Mike (Beacon Bay)
Date: October 2025
1. Overview
ck is a hybrid semantic + BM25 code search engine with blazing-fast delta indexing and optional MCP server mode.
The VS Code / Cursor extension brings ck’s command-line power and semantic awareness into the editor itself — combining:
TUI-style search interaction (fast keyboard UX, dense visual layout)
Rich semantic results (ranked by meaning, not just text)
Tight integration with code navigation and context (jump, peek, reindex)
It replaces the need to alt-tab to the terminal or separate UIs. Developers can query, browse, and open results entirely within VS Code or Cursor.
2. Goals & Non-Goals
Goals
Create a fast, tactile in-editor search experience that feels like a TUI but fits VS Code’s sidebar paradigm.
Provide two integration modes:
CLI Mode: spawn the local ck binary (ck --hybrid "query").
MCP Mode: connect to a running MCP server (ck --serve) for structured results, pagination, and streaming.
Support inline navigation (open file + jump to line/col).
Provide visual cues for index health and mode.
Expose command palette actions (Search, Search Selection, Reindex).
Deliver hybrid, semantic, or regex modes without breaking flow.
Make it Cursor-ready — leveraging MCP for richer, multi-tool integration.
Non-Goals (v1)
No team-shared search history or cloud index management.
No embedded LLM summaries or code rewriting.
No custom model tuning or embedding configuration via the extension (CLI handles that).
No multi-repo federation beyond the workspace root.
3. Target Users & Use Cases
Primary User:
Professional developer who prefers local tools over SaaS.
Works in large codebases with complex naming, needing both keyword and meaning search.
Already comfortable with rg, fzf, or ck CLI.
Secondary Users:
Cursor users who want an MCP-native experience for local or offline search.
OSS maintainers who want to quickly explore unfamiliar repos.
Core Use Cases:
“Find by meaning”
Developer types: “http server startup” → sees function calls like start_listener() or spawn_server().
“Search selection”
Highlights a function → runs ck semantic_search for similar patterns.
“TUI-like browsing”
Arrows navigate results, Enter jumps to code, Esc focuses query input.
“Instant reindex”
Runs ck reindex without leaving the editor; shows progress/status.
“Mode switch”
Quickly toggle between semantic, hybrid, and regex searches.
“Cursor integration”
MCP mode exposes structured search results for toolchains like Claude or Cursor’s code actions.
4. User Experience
4.1 Sidebar Layout
Top bar:
Input box for queries
Mode selector (Hybrid / Semantic / Regex)
Index status pill (green/yellow/red)
Body:
Scrollable list of results:
Each item: path:line + highlighted snippet
Click → opens in editor and jumps to location
Footer:
Toggle for showing raw CLI output (for power users)
4.2 Keyboard Interaction
Action	Keybinding	Description
Focus search bar	Ctrl+Shift+;	Opens sidebar and focuses query input
Search selection	Ctrl+Shift+'	Runs ck search on selected text
Navigate results	↑ / ↓	Move selection
Open selected result	Enter	Jump to file+line
Reindex	Command Palette (ck: Reindex)	Rebuilds index
4.3 Result Display
Results show filename, line number, and snippet.
Clicking a result:
Opens file.
Scrolls to line.
Briefly highlights the match.
4.4 Visual Style
Minimalist, terminal-inspired (monospaced, compact).
Uses VS Code theme colors.
Feels like a modern TUI: keyboard-centric, dense information, quick transitions.
5. Functional Requirements
5.1 Integration Modes
Mode	Mechanism	Description
CLI	Spawns ck binary with flags	Uses stdout/stderr to parse results
MCP	Starts or connects to ck --serve	Uses JSON-RPC to call hybrid_search, semantic_search, index_status, etc.
5.2 Commands
ck.search — Open sidebar and focus query input.
ck.searchSelection — Search current selection semantically.
ck.reindex — Trigger reindexing.
(Optional v1.1) ck.peekResults — Open “peek” view of matches inline.
5.3 Config Options
Setting	Type	Default	Description
ck.mode	string	cli	CLI or MCP mode
ck.cliPath	string	ck	Path to binary
ck.mcp.command	string	ck	MCP command
ck.mcp.args	array	[\"--serve\"]	Arguments for MCP server
ck.index.root	string	${workspaceFolder}	Root folder for indexing
ck.hybrid	boolean	true	Use hybrid search
ck.pageSize	number	50	Number of results per page
6. Technical Requirements
6.1 Language & Framework
Language: TypeScript
Target: VS Code extension API (>=1.93.0)
Frontend: WebviewViewProvider
Backend: Node.js child process + optional MCP client
6.2 Dependencies
jsonrpc-lite for MCP communication
typescript, @types/node, vsce for packaging
6.3 Performance
Search results should begin rendering <300ms after CLI output or MCP response.
No blocking UI on large outputs.
MCP connection should reconnect automatically on extension reload.
6.4 Security
Webview CSP: default-src 'none'; script-src 'nonce-*'; style-src 'unsafe-inline'.
No remote network calls.
Only interacts with local filesystem paths under workspace root.
7. Future Enhancements
Feature	Description	Priority
Streaming output	Show results as they’re emitted	High
Index health indicator	Call index_status periodically	High
Peek results	Inline “peek view” for matches	Medium
History + Re-run	Save last 10 queries	Medium
Workspace multi-root	Handle multiple ck roots	Medium
Result filters	File type / repo / path regex	Medium
Syntax highlighting	Render snippets with code coloration	Low
MCP discovery	Auto-detect running MCP server	Low
8. Success Metrics
Metric	Goal
Median search latency (local)	<500ms
Search results relevance satisfaction (user survey)	>80% positive
Adoption rate among existing ck users	50% of CLI users install extension
Cursor users connecting via MCP	≥25% of extension installs
Crash or hang rate	<1% of sessions
9. Release Plan
Phase 1: MVP (Weeks 1–3)
CLI mode search + sidebar UI.
Open-to-line navigation.
Command palette shortcuts.
Phase 2: MCP integration (Weeks 4–6)
JSON-RPC client integration.
Support hybrid_search and index_status.
Phase 3: UX polish (Weeks 7–8)
Add keyboard navigation.
Mode toggle + index status indicator.
Release v1.0 to Marketplace.
Phase 4: Cursor support (Weeks 9–10)
Register extension as MCP client.
Validate with Cursor team.
10. Competitive Landscape
Tool	Comparison
ripgrep / fzf	Fast keyword search, no semantic depth.
Sourcegraph / Cody	Semantic search via cloud, not local/offline.
Cursor MCP search	Great integration but not local or user-controlled.
ck (CLI)	Local, powerful, but CLI-only UX.
ck VS Code Extension	Merges ck’s power with editor-native discoverability.
11. Risks & Mitigation
Risk	Mitigation
MCP protocol changes	Keep versioned tool discovery & fallback to CLI mode.
Parsing CLI output	Standardize --json output mode for ck.
Large index performance	Paginate + stream results incrementally.
VS Code API changes	Use stable WebviewViewProvider interface.
12. Summary
This extension aims to bring ck’s semantic, hybrid, and regex power to the developer’s fingertips — directly inside the editor — with zero setup friction.
It’s fast, local, and pairs beautifully with ck’s philosophy: search code the way you think about it, not the way it’s written.
