import imaplib
import os
import ssl
import dotenv


def test_qq_imap_connection() -> None:

    _ = dotenv.load_dotenv()
    email_address = os.getenv("MAIL_ADDRESS")
    authorization_code = os.getenv("MAIL_AUTH_CODE")

    context = ssl.create_default_context()

    client: imaplib.IMAP4_SSL | None = None

    try:
        client = imaplib.IMAP4_SSL(
            host="imap.qq.com",
            port=993,
            ssl_context=context,
        )

        client.login(email_address, authorization_code)

        status, _ = client.select("INBOX", readonly=True)
        if status != "OK":
            raise RuntimeError("无法打开QQ邮箱收件箱")

        status, data = client.uid("search", None, "ALL")
        if status != "OK":
            raise RuntimeError("无法查询QQ邮箱邮件")

        message_uids = data[0].split()
        print(f"连接成功，收件箱共有 {len(message_uids)} 封邮件")

    finally:
        if client is not None:
            try:
                client.logout()
            except imaplib.IMAP4.error:
                pass


if __name__ == "__main__":
    test_qq_imap_connection()


