"""Framework-agnostic evaluation: build-order extraction and scoring metrics.

Deliberately imports nothing at package level. Both arms import the specific
module they need (`buildorder`, `metrics`) so that importing this package never
pulls an arm-specific, framework-bound reporting module into scope.
"""
