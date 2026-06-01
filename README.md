1、 下载依赖
pip install requirements.txt

2、运行程序
python -m uvicorn app.main:app --reload

更新记录
2026-06-01 fjy
将项目数据库连接方式调整为 MySQL，数据库地址通过 .env 中的 DATABASE_URL 配置。
新增用户注册、登录、退出登录功能。
新增 users 用户表，用于保存用户名和密码。
首页和主要功能页面增加登录校验，未登录用户会跳转到登录页面。
优化登录/注册页面样式。
