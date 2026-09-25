/**
 * ck VS Code Extension
 *
 * Brings semantic code search powered by ck directly into VS Code
 */

import * as vscode from 'vscode';
import { CkSearchPanel } from './searchPanel';
import { CkConfig } from './types';

let searchPanel: CkSearchPanel | undefined;

export function activate(context: vscode.ExtensionContext) {
  console.log('ck extension is now active');

  // Load configuration
  const config = loadConfig();

  // Register the webview view provider
  const provider = new CkSearchPanel(context.extensionUri, config);
  searchPanel = provider;

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      CkSearchPanel.viewType,
      provider,
      {
        webviewOptions: {
          retainContextWhenHidden: true
        }
      }
    )
  );

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('ck.search', async () => {
      // Focus the search view
      await vscode.commands.executeCommand('ck.searchView.focus');
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('ck.searchSelection', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
      }

      const selection = editor.selection;
      const text = editor.document.getText(selection);

      if (!text) {
        vscode.window.showWarningMessage('No text selected');
        return;
      }

      // Focus the search view and perform search
      await vscode.commands.executeCommand('ck.searchView.focus');

      if (searchPanel) {
        await searchPanel.search(text, config.defaultMode);
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('ck.reindex', async () => {
      if (searchPanel) {
        await searchPanel.reindex();
      }
    })
  );

  // Listen for configuration changes
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('ck')) {
        const newConfig = loadConfig();
        if (searchPanel) {
          searchPanel.updateConfig(newConfig);
        }
      }
    })
  );

  // Check if ck binary is available
  checkCkAvailability(config.cliPath);
}

export function deactivate() {
  if (searchPanel) {
    searchPanel.dispose();
    searchPanel = undefined;
  }
}

/**
 * Load configuration from VS Code settings
 */
function loadConfig(): CkConfig {
  const config = vscode.workspace.getConfiguration('ck');

  return {
    mode: config.get('mode', 'cli') as 'cli' | 'mcp',
    cliPath: config.get('cliPath', 'ck'),
    mcpCommand: config.get('mcp.command', 'ck'),
    mcpArgs: config.get('mcp.args', ['--serve']),
    indexRoot: config.get('index.root', '${workspaceFolder}'),
    defaultMode: config.get('defaultMode', 'semantic') as 'hybrid' | 'semantic' | 'regex',
    pageSize: config.get('pageSize', 25),
    threshold: config.get('threshold', 0.6),
    topK: config.get('topK', 50),
    contextLines: config.get('contextLines', 4)
  };
}

/**
 * Check if ck binary is available on PATH
 */
async function checkCkAvailability(cliPath: string) {
  const { spawn } = require('child_process');

  const child = spawn(cliPath, ['--version'], { shell: false });

  let available = false;

  child.on('close', (code: number) => {
    if (code === 0) {
      available = true;
    } else {
      vscode.window.showWarningMessage(
        `ck binary not found at "${cliPath}". Please install ck or update the "ck.cliPath" setting.`,
        'Install ck',
        'Open Settings'
      ).then((selection) => {
        if (selection === 'Install ck') {
          vscode.env.openExternal(vscode.Uri.parse('https://github.com/BeaconBay/ck#installation'));
        } else if (selection === 'Open Settings') {
          vscode.commands.executeCommand('workbench.action.openSettings', 'ck.cliPath');
        }
      });
    }
  });

  child.on('error', () => {
    vscode.window.showWarningMessage(
      `ck binary not found at "${cliPath}". Please install ck or update the "ck.cliPath" setting.`,
      'Install ck',
      'Open Settings'
    ).then((selection) => {
      if (selection === 'Install ck') {
        vscode.env.openExternal(vscode.Uri.parse('https://github.com/BeaconBay/ck#installation'));
      } else if (selection === 'Open Settings') {
        vscode.commands.executeCommand('workbench.action.openSettings', 'ck.cliPath');
      }
    });
  });
}
