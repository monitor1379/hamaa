# 开发 Development

Hamaa由作者[Zhenpeng Deng（monitor1379）](https://github.com/monitor1379)在2016年10月创立。目前主要由个人负责开发、维护与改进。

## 理念 Belief

Hamaa始终遵循着`too SIMPLE and sometimes NAIVE`的原则来设计：

- **简单**: 每一个可配置项都抽象成简单的模块。具体地，网络层、损失函数、优化器、初始化器、激活函数都是独立
的模块，易于使用、扩展与修改。

- **朴素**: 所有模块都使用朴素的代码实现。每一段源代码都希望能在第一次阅读时显得直观易懂，具有良好的可读性，
并且不过分使用trick。


## 报告 Bug

请您在[GitHub](https://github.com/monitor1379/hamaa)上提交Bug报告。具体需要包含以下内容: 

- Hamaa的版本号

- 运行平台（Windows、Ubuntu、macOS或者docker等等）

- 重现Bug的步骤

- 实际获得的结果与期望获得的结果

如果您不确定这是否是Bug，或者不确定是否与Hamaa相关，您可以[发邮件](mailto:yy4f5da2@hotmail.com)给作者说明情况。


## 修复 Bug

您可以通过GitHub的问题栏(issues)来查看Bug报告。

- 任何被标记为Bug的项对所有想要修复它的人来说都是开放的。

- 如果您发现了可以自己修复的amaa的一个Bug，
您可以用任何方法来实现修复并且无需立即报告这个Bug。


## 加入我们 Join us

如果您为Hamaa添加了新的扩展（比如网络层、损失函数、优化器、初始化器或者激活函数等等），或者将Hamaa构建了某种新的复杂模型，欢迎您分享您的代码给Hamaa。

- 说明它是怎么工作的，如果可以的话请给出学术论文的链接。

- 尽可能的给出完善的说明文档。

- 在GitHub通过pull request提交您的申请。

## 如何加入 How to join us

### 文档

- Hamaa使用[Markdown](http://wowubuntu.com/markdown/#list)编写代码注释与文档，

- 并使用[MkDocs](http://www.mkdocs.org/)生成文档。

- 代码托管在[Read the Docs](https://readthedocs.org/)上进行构建与版本管理。

如果你想重新生成整个文档，在Hamaa项目根目录（即`mkdocs.yml`文件同级目录下）运行以下命令:
```bash
$ mkdocs build
```

或者使用以下方法在浏览器中预览:
```bash
$ mkdocs serve
```

编写文档时，请尽可能地按照现有文档的文字习惯来保证整个库的文字风格一致性。
所使用的语法及约定的相关信息，请参考以下文档：

- TODO
- [A Guide to NumPy/SciPy Documentation](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)


### 测试


每当您更改任何代码的时候，您应该运行测试脚本来测试它是否能优化现有属性。

- Hamaa使用[nose](http://nose.readthedocs.io/en/latest/)进行本地单元测试。

- 使用[Travis-CI](https://travis-ci.org/)进行远程集成测试。

- 使用[coverage](https://coverage.readthedocs.io/en/coverage-4.2/) / [codecov](https://codecov.io/)进行本地 / 远程代码覆盖率检测。

编写测试case时，请尽可能地按照现有测试习惯来保证整个库的测试风格一致性。
所使用的语法及约定的相关信息，请参考以下文档：

- TODO

- TODO

## 发送拉请求 Pull request

当您对您添加的内容感到满意并且测试通过，文档规范，简明，不存在任何注释错误时，
您可以将您的更改提交到一个新的分支(branch)，并且将这个分支与您的副本(fork)合并，
然后通过GitHub的网站发送一个拉请求(pull request)。

所有的这些步骤在GitHub上有相当不错的说明: [https://guides.github.com/introduction/flow/](https://guides.github.com/introduction/flow/)

当您提交拉请求时，请附带一个更改内容的说明，以帮助我们能更好的检阅它。

如果它是一个正在开放的问题(issue)，比如：issue#123，
请在您的描述中添加 Fixes#123,*Resolves#123*或者*Closes#123*，
这样当您的拉请求被接纳之后GitHub会关闭那个问题。