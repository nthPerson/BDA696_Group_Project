# Vendored: uPlot 1.6.32

Fast canvas time-series charts, MIT licensed (see `LICENSE`). Source: <https://github.com/leeoniya/uPlot>.
Committed here so the viewer needs no npm install and works offline at a demo (ADR-0026).

| File | sha256 |
|---|---|
| `uPlot.iife.min.js` | `19c8d4c6ad88929a79f4ae49d6f7161566dfd0ba3d15cc495e974f787eb78f1f` |
| `uPlot.min.css` | `df630c6a8d6f8eeaff264b50f73ce5b114f646ffd9a0bb74f049b0a00135fa04` |

Fetched with `npm pack uplot@1.6.32` (tarball integrity checked against `npm view uplot@1.6.32
dist.integrity`), files copied from `package/dist/` and `package/LICENSE`.

## Bumping the version

1. In an empty temporary directory: `npm pack uplot@<version>`, then `tar -xzf uplot-<version>.tgz`.
2. Copy `package/dist/uPlot.iife.min.js`, `package/dist/uPlot.min.css` and `package/LICENSE` here.
3. Update the version, the hashes (`sha256sum uPlot.*`) and this README.
4. `make ble-viewer`, press **Simulate**: both charts must draw and scroll, with no console errors.
