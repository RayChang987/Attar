# Releasing

For the maintainers. A release is a version on PyPI, a git tag, and (when the data changed) the zoo on the Hub.

1. **Version.** Set `__version__` in `attar/__init__.py` (the only place it lives) and add a section to
   `CHANGELOG.md`. The Colab notebooks install a pinned version (`!pip install -q attar==<version>` in
   `examples/colab/*.ipynb`): pin the new one, and the README's tests badge to the count of step 2.
2. **Check.** `ruff check attar tests tools examples experiments` and `python -m unittest discover -s tests` (with the
   local zoo) both pass.
3. **Build.** `rm -rf dist && python -m build`, then `twine check dist/*`. The wheel must hold the C++
   sources built on first use: `attar/timing/csrc/{pdtree,dmp,activity}.cpp` and `attar/naive/csrc/abacus.cpp`
   (`unzip -l dist/*.whl | grep csrc`).
4. **Try it as a user.** Install the wheel into a clean environment (a fresh venv, or a `python:3.x` container) and
   run `attar.load` on a zoo design, `d.place()`, `d.hpwl()`, `d.timing_report()` and `d.export()`, without a
   Hugging Face token. Run the Colab notebooks from the wheel too (a GPU of 16 GB or less, as Colab's T4).
5. **TestPyPI first.** `twine upload --repository testpypi dist/*`, then
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ attar` in a clean
   environment.
6. **PyPI.** `twine upload dist/*`, then `git tag v<version> && git push origin v<version>`.
   On the first release, also refresh the zoo's card (it says `pip install attar`): `python tools/zoo_release.py`,
   then `hf upload RayChang987/attar-zoo zoo_hub/README.md README.md --repo-type dataset`.
7. **Zoo** (only when it changed): `python tools/zoo_release.py`, then
   `hf upload-large-folder RayChang987/attar-zoo zoo_hub/ --repo-type dataset`. On the Hub arrays and large JSON
   are zstd-compressed (`<file>.zst`; `attar.fetch` decompresses); the upload does not delete stale files there
   (`hf repo-files delete`).
