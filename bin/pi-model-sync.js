#!/usr/bin/env -S node --experimental-strip-types --no-warnings
import { join } from 'node:path';
import { homedir } from 'node:os';
import { syncModels } from '../extensions/model-sync.ts';

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run') || args.includes('-n') || args.includes('--preview');
const targetProviders = args.filter((arg) => !arg.startsWith('-'));

const agentDir = process.env.PI_CODING_AGENT_DIR || join(homedir(), '.pi', 'agent');

console.log(`[pi-model-sync] Mode: ${dryRun ? 'DRY-RUN (Preview only)' : 'LIVE SYNC'}`);
console.log(`[pi-model-sync] Target config: ${join(agentDir, 'models.json')}`);

try {
  const result = await syncModels({ dryRun, targetProviders, agentDir });

  console.log('\n--- Sync Results ---');
  for (const provider of result.providers) {
    const diff = result.diffs?.find((d) => d.provider === provider);
    const count = result.counts[provider] ?? 0;
    if (diff) {
      console.log(`✓ Provider [${provider}]: ${count} models (Added: ${diff.added}, Updated: ${diff.updated}, Unchanged: ${diff.unchanged})`);
      if (dryRun && diff.details.length > 0) {
        for (const item of diff.details) {
          if (item.type === 'added') {
            console.log(`    + [NEW] ${item.id}`);
          } else if (item.type === 'updated') {
            console.log(`    ~ [MOD] ${item.id} (${item.changes?.join(', ')})`);
          }
        }
      }
    } else {
      console.log(`✓ Provider [${provider}]: ${count} models`);
    }
  }

  if (result.skipped.length > 0) {
    console.log(`\n• Skipped providers: ${result.skipped.join(', ')}`);
  }

  if (result.errors.length > 0) {
    console.error('\n✗ Errors encountered:');
    for (const err of result.errors) {
      console.error(`  - ${err}`);
    }
  }

  if (dryRun) {
    console.log('\n[Dry-run] No files were modified.');
  } else {
    console.log(`\n[Success] models.json updated successfully (backup saved to models.json.bak).`);
  }
} catch (err) {
  console.error('\n[Fatal Error]:', err instanceof Error ? err.message : String(err));
  process.exit(1);
}
