# recast

## Commit messages

Every commit message is one line:

```
<type>: <what changed, in one sentence>
```

| Type | Use it for | Example |
|---|---|---|
| `init:` | Setting up a repo, project or module for the first time | `init: Initial commit` |
| `feat:` | Adding or extending functionality | `feat: depth buffer that is cleared each frame and recreated with the swapchain` |
| `fix:` | Correcting a bug or broken behaviour | `fix: swapchain recreate no longer hangs when the window is closed while minimized` |

Rules:

- Lowercase type, then a colon and a space (`feat: `, not `Feat:` or `feat -`).
- Describe the change, not the process (`feat: glTF model loading`, not `feat: worked on loader`).
- One sentence, no trailing period.
- One logical change per commit. If the message needs "and also", split the commit.
