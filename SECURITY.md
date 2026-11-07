# Security Policy

Autonomous Agent Lab is an open-source educational project under [Tiny Robot Lab](https://tinyrobotlab.com). Tiny Robot Lab itself is currently operated as a non-commercial learning project. The repository code is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0-only), and the license text governs permitted use.

## Supported Versions

Security fixes and updates are applied exclusively to the default `main` branch, which is actively maintained. Tagged releases represent historical learning milestones and do not receive guaranteed security backports.

| Version / Branch | Supported          | Description                                                       |
| :--------------- | :----------------- | :---------------------------------------------------------------- |
| `main`           | :white_check_mark: | Actively maintained branch receiving current fixes                |
| Tagged releases  | :x:                | Historical learning milestones; no guaranteed security backports    |

## Reporting a Vulnerability

We appreciate responsible disclosure of security vulnerabilities.

- **Do NOT file a public issue:** Please do not disclose vulnerabilities, potential security flaws, or sensitive exploits in public GitHub issues, pull requests, or discussions.
- **Preferred Reporting Method:** Please report vulnerabilities privately using GitHub's **Private Vulnerability Reporting**:
  1. Navigate to the repository's **Security** tab.
  2. Under "Reporting", select **Advisories**, then click **Report a vulnerability**.
  3. Provide a clear description of the issue, affected components, steps to reproduce, and any suggested mitigation.
- **Alternative Contact:** If private vulnerability reporting is not yet available on the repository, hold the disclosure and reach out privately to the repository maintainer through the contact information listed on their GitHub profile ([@tomhalsan](https://github.com/tomhalsan)).

### Scope: Vulnerabilities vs. Ordinary Bugs

- **Security Vulnerabilities:** Issues that lead to credential leakage, unauthorized host or container access, remote code execution, denial of service on shared infrastructure, or bypassing security boundaries.
- **Ordinary Bugs:** Architectural quirks, heuristic ranking mistakes, formatting flaws, or local test failures that do not pose an active security threat. These should be reported as standard public GitHub issues.

## Credential & Secret Hygiene

This repository never intentionally stores or distributes private credentials, API tokens, or secrets.

If you believe an API key (such as a Tavily API key, Cloudflare token, Cloudflare Tunnel credential, or SSH key) was inadvertently committed or exposed:
1. **Revoke and rotate immediately:** Treat the credential as compromised at the external provider immediately.
2. Deleting a commit or removing the file in a subsequent commit does not guarantee the secret was not harvested. Full revocation and rotation at the provider is mandatory.

