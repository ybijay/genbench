# Your test packs

Put each pack in its own folder: `packs/<name>/pack.json`. genbench loads every `pack.json` under the
folders listed in `[packs].dirs` of `genbench.toml` (this folder by default), next to the tests that ship with
genbench.

The quickest start is `genbench new-test`, which asks for a name, description, goal and expectations and
writes the test for you. Then run `genbench validate`.
