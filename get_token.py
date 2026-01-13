from fyers_apiv3 import fyersModel

client_id = "ZKX7NS29YX-100"
secret_key = "SC26Y9FJLX"
redirect_uri = "https://127.0.0.1:5000/"
auth_code = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJaS1g3TlMyOVlYIiwidXVpZCI6IjBlYmM4ODQ2ODk1ZDRhM2Q4YmMxNWIxNmU5YTRiZTUxIiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IkZBSDg1NDI0Iiwib21zIjoiSzEiLCJoc21fa2V5IjoiMGI0YWQ1MmVjYWZjNGVhYzE2OTY1MmI5NGMzMDlhNTg5ODU3NDVjYWMzZWEyYTlhYjA2MmUzYTciLCJpc0RkcGlFbmFibGVkIjoiTiIsImlzTXRmRW5hYmxlZCI6Ik4iLCJhdWQiOiJbXCJkOjFcIixcImQ6MlwiLFwieDowXCIsXCJ4OjFcIixcIng6MlwiXSIsImV4cCI6MTc2ODE2MjQzMiwiaWF0IjoxNzY4MTMyNDMyLCJpc3MiOiJhcGkubG9naW4uZnllcnMuaW4iLCJuYmYiOjE3NjgxMzI0MzIsInN1YiI6ImF1dGhfY29kZSJ9.oG5tgNmywiQeccLDQzG6IbmnqO55-VuGuLNibNTzATw"

session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type="code",
    grant_type="authorization_code"
)

session.set_token(auth_code)
response = session.generate_token()

print("Full response:")
print(response)
