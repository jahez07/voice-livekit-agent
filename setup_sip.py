import asyncio
import os

from dotenv import load_dotenv
from livekit import api

load_dotenv(".env")
IP_ADDRESS = os.getenv("IP_ADDRESS")

async def setup_sip():
    lk = api.LiveKitAPI(
        url=f"ws://{IP_ADDRESS}:7880",
        api_key="APIuoFDt5njRYLA",
        api_secret="MooeqeW9ae5QvvlgjFFGbuFkzA7160g9e5mSVPfAbEIH",
    )

    # Check existing trunks
    trunks = await lk.sip.list_sip_inbound_trunk(api.ListSIPInboundTrunkRequest())
    rules = await lk.sip.list_sip_dispatch_rule(api.ListSIPDispatchRuleRequest())

    if trunks.items or rules.items:
        print("═" * 50)
        print("Existing SIP configuration found:")
        print("═" * 50)

        if trunks.items:
            print(f"\nInbound trunks ({len(trunks.items)}):")
            for t in trunks.items:
                print(f"  ID:       {t.sip_trunk_id}")
                print(f"  Name:     {t.name}")
                print(f"  Numbers:  {list(t.numbers)}")
                print(f"  Allowed:  {list(t.allowed_addresses)}")
                print()

        if rules.items:
            print(f"Dispatch rules ({len(rules.items)}):")
            for r in rules.items:
                print(f"  ID:   {r.sip_dispatch_rule_id}")
                print(f"  Name: {r.name}")
                print()

        print("Skipping creation. Delete existing config first to recreate.")
        await lk.aclose()
        return

    # No existing config — create trunk
    trunk = api.SIPInboundTrunkInfo(
        name="3CX",
        allowed_addresses=["192.168.2.185"],
        numbers=["245"],
    )
    await lk.sip.create_sip_inbound_trunk(api.CreateSIPInboundTrunkRequest(trunk=trunk))
    print("✅ Inbound trunk created")

    # Create dispatch rule
    rule = api.SIPDispatchRuleInfo(
        name="Agent-Dispatch",
        rule=api.SIPDispatchRule(
            dispatch_rule_individual=api.SIPDispatchRuleIndividual(room_prefix="call-")
        ),
    )
    await lk.sip.create_sip_dispatch_rule(api.CreateSIPDispatchRuleRequest(rule=rule))
    print("✅ Dispatch rule created")

    await lk.aclose()


asyncio.run(setup_sip())
