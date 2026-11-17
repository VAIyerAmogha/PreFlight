You get a 404 because the site has never been deployed. site/DEMO_DEV.md still says "Not deployed", and no Vercel project exists yet. When nothing is deployed at a *.vercel.app address, Vercel just returns a 404. The site itself is fine: I ran npm run build in site/ and it built cleanly as a static export with two pages, / and /try.

Do it in this order: deploy the site, put the link in the README, then publish to PyPI. PyPI shows the README exactly as it was at upload time. If you upload first and add the link later, the link won't appear on PyPI until 2.0.1.

1. Push your local commits

Your local v2-dev is 5 commits ahead of GitHub, so push it and merge to main as usual. Vercel deploys from GitHub, so it needs the site/ code, and site/public/wheels/ in particular, to be there.

2. Create the Vercel project

1. Go to vercel.com → Add New → Project and import VAIyerAmogha/PreFlight.
2. Project name: pypreflight. This name is what gives you pypreflight.vercel.app. If someone else already has it, Vercel will suggest something like pypreflight-xyz. Use that name, or add a custom domain.
3. Root Directory: site. This step matters most: the Next.js app is in that folder, not at the repo root.
4. Leave Framework (Next.js), Build and Output at their defaults, set Production Branch to main, and click Deploy.
5. Open https://pypreflight.vercel.app and /try, then load a sample to check that the in-browser engine installs.

3. Add the link to the README and pyproject

- Put the absolute URL https://pypreflight.vercel.app near the top of README.md.
- Add this to pyproject.toml:
[project.urls]
Demo = "https://pypreflight.vercel.app"
Repository = "https://github.com/VAIyerAmogha/PreFlight"

4. Publish to PyPI

Follow the steps from before: rm -rf dist && python -m build && twine check dist/* && twine upload dist/*.

5. After the release, point the demo at PyPI

For now the demo installs its own bundled pypreflight-2.0.0.dev0 wheel, set in site/lib/config.ts:10. Once 2.0.0 is on PyPI, change that line to:
export const PREFLIGHT_SPEC = "pypreflight==2.0.0";
Then push, and Vercel redeploys. The demo works with the dev0 wheel in the meantime, so this can wait.

I can add the README link and [project.urls] now, before you deploy. If the project name turns out to be taken in step 2, we just change the URL.



Right now the package is still on 2.0.0.dev0 in two places, and the dist/ folder holds dev0 builds. Nothing in the repo publishes automatically: ci.yml has no release job. Here are the steps to do it by hand:

1. Change the version to 2.0.0
- pyproject.toml: version = "2.0.0"
- src/preflight/__init__.py:9: __version__ = "2.0.0"
- CHANGELOG.md: rename ## Unreleased to ## 2.0.0 (2026-11-16)
- Optional: the classifier still says Development Status :: 3 - Alpha. Change it to 4 - Beta or 5 - Production/Stable if that fits better.

2. Merge to main and tag
git commit -am "release: 2.0.0"
# merge v2-dev into main (via PR or locally), then on main:
git tag v2.0.0
git push origin main --tags

3. Build a clean release
rm -rf dist/
python -m pip install --upgrade build twine
- Optional: the classifier still says Development Status :: 3 - Alpha. Change it to 4 - Beta or 5 - Production/Stable if that fits better.

2. Merge to main and tag
git commit -am "release: 2.0.0"
# merge v2-dev into main (via PR or locally), then on main:
git tag v2.0.0
git push origin main --tags

3. Build a clean release
rm -rf dist/
python -m pip install --upgrade build twine
python -m build          # creates dist/pypreflight-2.0.0.tar.gz and the .whl
twine check dist/*
Delete the old dist files first. If you don't, twine upload dist/* will try to upload the dev0 files as well.

4. Optional: test on TestPyPI first
twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pypreflight==2.0.0

5. Upload to PyPI
twine upload dist/*
When it asks for credentials, the username is __token__ and the password is an API token from pypi.org → Account settings → API tokens. Scope the token to the pypreflight project.

PyPI doesn't let you re-upload a version number, even after you delete it. Run the tests and do step 4 before step 5.

If you want, I can make the version bump (step 1) and run the build and twine check (step 3) now. I'll leave the upload to you because it needs your token. I could also add a GitHub Actions workflow that publishes to PyPI whenever you push a v* tag, using PyPI's trusted publishing so you don't have to store a token.