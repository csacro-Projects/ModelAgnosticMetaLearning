# gym-twoDNavigation
These environments replicate the 2D Navigation environments used in the MAML paper (https://arxiv.org/abs/1703.03400) for use with Spinning Up by OpenAI.

## Install
Execute the following in order to install the environment:
`python3 -m pip install -e gym-twoDNavigation`

To be on the save side, we here list the version of all installed libraries that we used
- gym==0.15.7
- numpy==1.18.5
- matplotlib==3.1.1

## Sources
The structure/requirements for the environment was taken from the OpenAI gym documentation: https://github.com/openai/gym/blob/5e43d10e2521c6405773b34090a63248124a0177/docs/creating_environments.md  

The implementation of the environments themselves were adapted from
https://github.com/cbfinn/maml_rl/tree/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples (point_env_randgoal.py and point_env_randgoal_oracle.py respectively),
which is licensed under the following MIT License:

```
Copyright (c) 2016 rllab contributors

rllab uses a shared copyright model: each contributor holds copyright over
their contributions to rllab. The project versioning records all such
contribution and copyright details.
By contributing to the rllab repository through pull-request, comment,
or otherwise, the contributor releases their content to the license and
copyright terms herein.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The necessary modifications to the Spinning Up spinup/run.py implementation in order to be able to use custom/non-standard environments
were taken from the yet unmerged pull request https://github.com/openai/spinningup/pull/201/commits/7d9bf51c2983b22c814236ec91b305aaa70a52e9