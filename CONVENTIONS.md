# CONVENTIONS.md

## Naming Conventions

### Files and Modules
- snake_case for file names
- modules grouped by functionality (data, models, pruning, etc.)

### Classes
- PascalCase
- Example: GraphSAGEModel, BasePruner

### Functions
- snake_case
- Use verbs: train_model, load_dataset

### Variables
- snake_case
- Avoid abbreviations unless standard (e.g., num_epochs)

---

## Code Structure Rules

- Each module must have a single responsibility
- Avoid circular imports
- Keep public APIs clean and documented
- Use registries for extensibility

---

## Config Rules

- No hardcoded hyperparameters
- Everything configurable via YAML
- Provide defaults via presets

---

## Logging

- Use consistent logging format
- Avoid print statements in core modules

---

## Error Handling

- Fail fast with clear error messages
- Validate inputs early

---

## Documentation

- Every public function/class must have a docstring
- Complex logic must include inline comments

---

## Testing

- Each module must have at least one test
- Use small datasets for tests (Citeseer)

---

## Performance

- Prioritize correctness first
- Optimize only after pipeline works
