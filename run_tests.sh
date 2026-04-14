# run all tests python files
#!/bin/bash
# run all test files in the current directory
for test_file in tests_*.py; do
    echo "Running $test_file..."
    python "$test_file"
done