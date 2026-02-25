### Recommended Use Cases

- Fast region matching when target appearance is relatively stable.
- UI element or marker tracking in controlled scenes.

### Minimal Run Example

```bash
pixi run -e default python scripts/run_cpp_service.py --service-dir services/f8/cvkit/template_match --exe f8cvkit_template_match_service --env-var F8CVKIT_TEMPLATE_MATCH_EXE
```

### Common Pitfalls

- Poor template initialization leads to drift.
- Significant scale/rotation changes reduce matching confidence.

### Troubleshooting

- Reinitialize template when confidence drops.
- Restrict search area if scene context is known.
