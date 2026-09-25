// Color palette - using RGB for consistency across terminals
use ratatui::style::Color;

pub const COLOR_CYAN: Color = Color::Rgb(80, 200, 200); // Cyan - headers, highlights
pub const COLOR_YELLOW: Color = Color::Rgb(255, 220, 0); // Yellow - matched lines, commands
pub const COLOR_WHITE: Color = Color::Rgb(220, 220, 220); // White - primary text
pub const COLOR_DARK_GRAY: Color = Color::Rgb(100, 100, 100); // Dark gray - subtle text
pub const COLOR_GRAY: Color = Color::Rgb(150, 150, 150); // Gray - secondary text
pub const COLOR_GREEN: Color = Color::Rgb(80, 200, 120); // Green - success, chunk boundaries
pub const COLOR_MAGENTA: Color = Color::Rgb(200, 80, 200); // Magenta - special markers
pub const COLOR_BLACK: Color = Color::Rgb(0, 0, 0); // Black - backgrounds

// Enhanced chunk colors for better visualization
pub const COLOR_CHUNK_HIGHLIGHT: Color = Color::Rgb(255, 165, 0); // Orange - highlighted chunk
pub const COLOR_CHUNK_BOUNDARY: Color = Color::Rgb(0, 255, 127); // Spring green - chunk boundaries
pub const COLOR_CHUNK_TEXT: Color = Color::Rgb(255, 255, 255); // Bright white - highlighted chunk text
pub const COLOR_CHUNK_LINE_NUM: Color = Color::Rgb(255, 215, 0); // Gold - highlighted chunk line numbers

pub const SPINNER_FRAMES: [char; 4] = ['|', '/', '-', '\\'];
pub const DEBOUNCE_MS: u64 = 300;
