# Testing Order Reference

Default order:

1. format / lint
2. type checks / static analysis
3. unit tests
4. smoke tests
5. integration tests
6. environment checks
7. live DB validation
8. deploy validation

Guidelines:
- start low-risk
- expand only as needed
- do not run destructive or ambiguous live actions without escalation
- report both failures and untested gaps clearly