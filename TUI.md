# ck TUI (Terminal User Interface)

The ck TUI provides an interactive search interface with real-time results, multiple preview modes, and keyboard-driven navigation.

## Quick Start

```bash
# Launch TUI in current directory
ck --tui

# Launch with initial query
ck --tui "error handling"

# Launch in specific directory
ck --tui --path /path/to/code
```

## Keyboard Shortcuts

### Navigation
| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate through search results |
| `PageUp` / `PageDown` | Scroll preview (in full-file mode) |
| `Enter` | Open selected file(s) in `$EDITOR` |
| `Ctrl+Up` / `Ctrl+Down` | Navigate search history |

### Search & Modes
| Key | Action |
|-----|--------|
| `Tab` | Cycle search modes (Semantic → Regex → Hybrid) |
| Type any text | Update search query (300ms debounce) |
| `Backspace` | Delete character from query |
| `/command` | Enter command mode (see Commands below) |

### View Controls
| Key | Action |
|-----|--------|
| `Ctrl+V` | Cycle preview modes (Heatmap → Syntax → Chunks) |
| `Ctrl+F` | Toggle snippet/full-file view |
| `Ctrl+D` | Show chunk metadata for current file |

### Multi-Select
| Key | Action |
|-----|--------|
| `Ctrl+Space` | Toggle selection of current file |
| `Enter` | Open all selected files (or current if none selected) |

### Exit
| Key | Action |
|-----|--------|
| `q` or `Esc` | Quit TUI |
| `Ctrl+C` | Force quit |

## Search Modes

### Semantic Search (Default)
Finds code by meaning using embeddings. Best for conceptual queries:
```
"error handling"
"database connection pooling"
"user authentication"
```

### Regex Search
Traditional pattern matching for exact text searches:
```
"TODO|FIXME"
"function.*Error"
"\bauth\b"
```

### Hybrid Search
Combines semantic understanding with keyword precision using Reciprocal Rank Fusion:
```
"async timeout"
"cache invalidation"
```

## Preview Modes

### Heatmap Mode (Default)
Shows semantic similarity with color-coded highlighting:
- **Red**: Lower similarity (0.6-0.7)
- **Yellow**: Medium similarity (0.7-0.85)
- **Green**: High similarity (0.85+)

### Syntax Mode
Displays syntax-highlighted code using your file's language:
- Powered by `syntect` for accurate highlighting
- Supports 7+ languages plus text formats

### Chunks Mode
Shows chunk boundaries and metadata:
- Visual indicators for chunk start/end
- Chunk type annotations (Function, Class, Method)
- Useful for understanding how code is indexed

## View Options

### Snippet View (Default)
Shows ±5 lines of context around matches. Perfect for quick scanning of results.

### Full File View
Press `Ctrl+F` to toggle. Features:
- Scrollable full file content
- Auto-scrolls to matched line when navigating results
- `PageUp`/`PageDown` for navigation
- Matched line stays highlighted

## Command Mode

Start your query with `/` to enter command mode:

```
/open <filename>    - Open specific file
/config             - Show current configuration
/help               - Show help message
```

## Multi-Select Workflow

1. Navigate to a file you want to open (`↑` / `↓`)
2. Press `Ctrl+Space` to select it
3. Continue selecting additional files
4. Press `Enter` to open all selected files

Selected files are shown with a `[✓]` indicator in the results list.

## Editor Integration

The TUI opens files in your `$EDITOR` (or `$VISUAL`) with line numbers. Supported editors:

| Editor | Support | Format |
|--------|---------|--------|
| Vim/Neovim | ✅ Full | Opens in tabs with line numbers |
| VS Code | ✅ Full | `-g file:line` format |
| Cursor | ✅ Full | `-g file:line` format |
| Sublime | ✅ Full | `file:line` format |
| Emacs | ✅ Limited | Opens first file only |
| Nano | ✅ Limited | Opens first file only |

Set your editor:
```bash
export EDITOR=vim
export EDITOR=code
export EDITOR=cursor
```

## Progress Tracking

The TUI shows detailed progress during indexing:

```
Indexing repository for semantic search...
src/main.rs • 23/145 files • 5/12 chunks
[████████████░░░░░░░░] 67%
```

- **File name**: Currently processing file
- **File progress**: Completed/total files
- **Chunk progress**: Completed/total chunks in current file
- **Progress bar**: Overall completion percentage

## Configuration

TUI preferences are automatically saved to:
- **Linux/macOS**: `~/.config/ck/tui.json`
- **Windows**: `%APPDATA%\ck\tui.json`

Saved settings:
- Last used search mode
- Preview mode preference
- Full-file mode setting

## Search History

The TUI maintains a history of your last 20 searches:
- Navigate with `Ctrl+Up` / `Ctrl+Down`
- History persists across searches within same session
- Duplicate queries are not added

## Performance Tips

1. **First search**: May take longer as index is built
2. **Subsequent searches**: Near-instant using cached index
3. **Large codebases**: Use specific query terms to reduce result set
4. **Snippet mode**: Faster rendering than full-file mode

## Troubleshooting

### TUI won't start
- Ensure terminal supports colors: `echo $TERM`
- Check terminal size: `stty size` (minimum 80x24)

### Editor doesn't open
- Verify `$EDITOR` is set: `echo $EDITOR`
- Ensure editor is in `$PATH`

### No search results
- Check search mode (Tab to cycle)
- Try broader query terms
- Verify files are indexed: `ck --status .`

### Slow performance
- First search builds index (one-time cost)
- Large result sets: Add threshold with `--threshold 0.7`
- Check index size: `du -sh .ck/`

## Examples

### Find authentication code
```bash
ck --tui "authentication"
# Press Tab to cycle to Regex mode
# Type: "login|auth|credentials"
```

### Review error handling
```bash
ck --tui "error handling"
# Use Ctrl+Space to select multiple files
# Press Enter to open all in editor
```

### Explore code structure
```bash
ck --tui "database"
# Press Ctrl+V to switch to Chunks mode
# Press Ctrl+F for full-file view
# Use PageDown to explore
```

## Architecture

The TUI is implemented in the `ck-tui` crate:
- **app.rs** (988 lines): Main event loop and state management
- **preview.rs** (658 lines): Preview rendering (heatmap, syntax, chunks)
- **chunks.rs** (428 lines): Chunk display and metadata
- **commands.rs** (408 lines): Command mode execution
- **rendering.rs** (210 lines): UI component rendering
- **state.rs** (45 lines): Application state
- **config.rs** (90 lines): Configuration persistence

Built with [ratatui](https://github.com/ratatui-org/ratatui) and [crossterm](https://github.com/crossterm-rs/crossterm).

## Contributing

The TUI is under active development. Contributions welcome for:
- Additional preview modes
- Custom keyboard shortcuts
- Theme/color customization
- Performance optimizations

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.
