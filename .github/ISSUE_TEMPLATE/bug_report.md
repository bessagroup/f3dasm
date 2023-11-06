name: "\U0001F41BBug report"
description: Create a report to help us improve f3dasm.
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report! Please write a clear and concise description of what the bug is.
  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected behavior
      description: Please write a clear and concise description of what you expected to happen.
    validations:
      required: false
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: |
        Please give the version of f3dasm that your are using.
    validations:
      required: false
  - type: textarea
    id: logs
    attributes:
      label: Error messages, stack traces, or logs
      description: Please copy and paste any relevant error messages, stack traces, or log output.
      render: shell
    validations:
       required: false