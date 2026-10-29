# Contributing

Use Node.js 20+ and Python 3. No npm installation is required to run the core app or tests.

```sh
npm run check
npm test
python3 -m http.server 8000 --directory dist
```

For algorithm changes, add a regression case for the behavior being fixed and describe its effect on existing datasets. Keep geometry and evaluation independent from browser APIs. For UI changes, check desktop and narrow viewports and keyboard operation; include actual verification limits in the change description.

Never report synthetic test metrics as real-user accuracy. Keep train/test boundaries explicit. Do not commit participant data, camera captures or secrets. Discuss any new dependency, external request or data persistence mechanism in the pull request.
