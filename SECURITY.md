# Security policy

## Supported version

Security fixes are applied to the latest commit on the default branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. If that option is unavailable, contact the repository owner privately through the contact method on the GitHub profile. Do not open a public issue containing credentials, exploit details, or private data.

Include the affected commit, reproduction steps, impact, and any suggested mitigation. You should receive an acknowledgement within seven days.

## Credential and data boundary

The offline demo uses local synthetic snapshots and requires no credentials or network access. Keep API keys, tokens, `.env` files, and private user data outside the repository. If a credential is exposed, revoke it at the provider before removing it from Git history.

Only use data you are authorized to collect, process, and redistribute. The checked-in fixtures are fully synthetic.
