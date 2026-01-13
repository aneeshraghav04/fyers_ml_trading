from fyers_apiv3 import fyersModel
import webbrowser

client_id = "ZKX7NS29YX-100"
redirect_uri = "https://127.0.0.1:5000/"
response_type = "code"
state = "sample_state"

session = fyersModel.SessionModel(
    client_id=client_id,
    redirect_uri=redirect_uri,
    response_type=response_type,
    state=state
)

auth_url = session.generate_authcode()
print("Open this URL in browser:", auth_url)
webbrowser.open(auth_url)
