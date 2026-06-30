# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

**Repo note:** `wontfix` already exists in `bessagroup/f3dasm` and maps directly. The other four (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`) are not yet defined as GitHub labels — `/triage` will create them on first use. If you later prefer to reuse adjacent existing labels (e.g. `question` for `needs-info`, `help wanted` for `ready-for-human`), edit the right-hand column instead.

Edit the right-hand column to match whatever vocabulary you actually use.
