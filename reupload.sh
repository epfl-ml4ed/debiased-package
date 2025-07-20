python3 -m pip install --upgrade build
rm -rf dist/
python3 -m build
twine check dist/*
python3 -m pip install --upgrade twine
python3 -m twine upload dist/*

