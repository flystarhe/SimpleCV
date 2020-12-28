# SimpleCV

Python 3.7+.

## Git Notes

* [Git 分支 - 分支的新建与合并](https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%9A%84%E6%96%B0%E5%BB%BA%E4%B8%8E%E5%90%88%E5%B9%B6)
* [Git 分支 - 分支开发工作流](https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E5%BC%80%E5%8F%91%E5%B7%A5%E4%BD%9C%E6%B5%81)

## Sphinx
* `cd docs`进入文档目录
* `sphinx-quickstart`初始化
* 在`docs/source/conf.py`中完善配置
* 生成API文档`sphinx-apidoc -o source ../`
* 生成文档`make html`
* 清理`make clean`

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

...

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
```
