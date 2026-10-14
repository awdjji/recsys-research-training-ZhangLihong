问题：git clone https://github.com/OrchidViolet/FatsMB.git \
fatal: 无法访问 'https://github.com/OrchidViolet/FatsMB.git/'：GnuTLS recv error (-110): The TLS connection was non-properly terminated.\
解决方案：Git 可能默认使用了不兼容的 TLS 协议版本，强制指定 TLS 1.2\
git -c http.sslVersion=tlsv1.2 clone https://github.com/OrchidViolet/FatsMB.git

问题：LightningCLI 依赖 jsonargparse 做签名解析，装完 PL 后需要额外装 `jsonargparse[signatures]`\
解决方案:先检查PL的版本，再安装对应的版本\