# 如何在 Mac 上将文件上传到 GitHub

以下步骤演示如何在 **macOS** 环境下，通过命令行将本地文件上传到 GitHub 仓库。

---

## 1. 安装 Git
Mac 一般自带 Git，可以通过以下命令检查：
```bash
git --version
```
如果未安装，运行：
```bash
xcode-select --install
```

---

## 2. 配置 Git（首次使用）
设置用户名和邮箱（会记录在提交历史里）：
```bash
git config --global user.name "YourName"
git config --global user.email "youremail@example.com"
```

---

## 3. 克隆远程仓库到本地
在 GitHub 创建仓库，复制仓库地址（HTTPS 或 SSH），然后运行：
```bash
git clone https://github.com/username/repository.git
cd repository
```

---

## 4. 将文件放到仓库目录
把要上传的文件复制或移动到该仓库的文件夹中。

---

## 5. 添加文件到暂存区
```bash
git add .
```
（`.` 表示添加所有修改过的文件）

---

## 6. 提交更改
```bash
git commit -m "Add project files"
```
请将 `"Add project files"` 替换为有意义的提交信息。

---

## 7. 推送到 GitHub
```bash
git push origin main
```
> 如果仓库默认分支是 `master`，请改为：
> ```bash
> git push origin master
> ```

首次推送时会要求验证身份：  
- **HTTPS 方式** → 使用 GitHub 用户名 + **Personal Access Token (PAT)**（不是账号密码）。  
- **SSH 方式** → 先配置 SSH key 并添加到 GitHub。

---

✅ 完成以上步骤后，你的文件就成功上传到 GitHub 仓库了！
