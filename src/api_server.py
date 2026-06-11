import asyncio
import os

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from livekit import api
from livekit.protocol.agent_dispatch import RoomAgentDispatch

load_dotenv(".env")

app = Flask(__name__)
CORS(app)

loop = asyncio.new_event_loop()


# Token endpoint (for the React frontend)
@app.route("/getToken", methods=["GET"])
def get_token():
    room = request.args.get("room", "my-room")
    identity = request.args.get("identity", "user")

    token = (
        api.AccessToken(
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET"),
        )
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
            )
        )
        .with_room_config(
            api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name="my-agent")],
            )
        )
    )

    return jsonify({"token": token.to_jwt(), "url": os.getenv("LIVEKIT_URL")})


# Twilio webhook (for inbound phone calls)
@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    from_number = request.form.get("From", "unknown")
    call_sid = request.form.get("CallSid", "unknown")

    async def get_connect_url():
        lkapi = api.LiveKitAPI()
        response = await lkapi.connector.connect_twilio_call(
            api.ConnectTwilioCallRequest(
                twilio_call_direction=api.ConnectTwilioCallRequest.TWILIO_CALL_DIRECTION_INBOUND,
                room_name=f"call-{call_sid}",
                participant_identity=from_number,
                participant_name=from_number,
                agents=[RoomAgentDispatch(agent_name="my-agent")],
            )
        )
        await lkapi.aclose()
        return response.connect_url

    connect_url = loop.run_until_complete(get_connect_url)

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Connecting you now.</Say>
    <Connect>
        <Stream url="{connect_url}" />
    </Connect>
</Response>"""

    return Response(twiml, mimetype="application/xml")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
