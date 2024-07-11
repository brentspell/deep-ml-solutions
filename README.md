# Deep-ML Solutions
This repo contains my solutions to the ML problems on [deep-ml.com](https://www.deep-ml.com/).

## Status
[![Coveralls](https://coveralls.io/repos/github/brentspell/deep-ml-solutions/badge.svg?branch=main)](https://coveralls.io/github/brentspell/deep-ml-solutions?branch=main)

## Development

### Setup
The following script creates a virtual environment using
[pyenv](https://github.com/pyenv/pyenv) for the project and installs
dependencies with [uv](https://pypi.org/project/uv/).

```bash
pyenv install 3.12
pyenv virtualenv 3.12 deep-ml
bin/deps
```

You can also use [pre-commit](https://pre-commit.com/) with the project to
run tests, etc. at commit time.

```bash
pre-commit install
```

### Testing
Testing, formatting, and static checking can all be done with pre-commit at
any time.

```bash
pre-commit run --all-files
```

There is also a watcher script that can be used to run these whenever a file
changes.

```bash
bin/watch
```

## License
Copyright © 2024 Brent M. Spell

Licensed under the MIT License (the "License"). You may not use this
package except in compliance with the License. You may obtain a copy of the
License at

    https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
