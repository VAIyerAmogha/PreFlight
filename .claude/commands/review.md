Use the reviewer agent on all files changed since the last git commit.

First run: git diff --name-only HEAD
to get the list of changed files.

Review each file. Return findings grouped by severity:
### Must fix
### Should fix
### Consider
### Tests missing

If the diff is empty, say so and stop.
