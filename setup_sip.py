import asyncio
from livekit import api


EXTENSIONS = [
    {
        "number": "8000",
        "room_prefix": "call-en-",
        "rule_name": "English Dispatch",
    },
    {
        "number": "8001",
        "room_prefix": "call-fr-",
        "rule_name": "French Dispatch",
    },
]


async def setup_sip():
    lk = api.LiveKitAPI(
        url="ws://10.1.114.7:7880",
        api_key="APIuoFDt5njRYLA",
        api_secret="MooeqeW9ae5QvvlgjFFGbuFkzA7160g9e5mSVPfAbEIH",
    )

    try:
        trunks = await lk.sip.list_sip_inbound_trunk(api.ListSIPInboundTrunkRequest())
        rules = await lk.sip.list_sip_dispatch_rule(api.ListSIPDispatchRuleRequest())

        print(f"Existing trunks: {len(trunks.items)}")
        print(f"Existing rules: {len(rules.items)}")

        # Create a single inbound trunk that accepts both extensions
        if not trunks.items:
            trunk = api.SIPInboundTrunkInfo(
                name="3CX Local",
                allowed_addresses=["10.1.3.1/32", "192.168.2.185"],
                numbers=[ext["number"] for ext in EXTENSIONS],
            )
            created = await lk.sip.create_sip_inbound_trunk(
                api.CreateSIPInboundTrunkRequest(trunk=trunk)
            )
            trunk_id = created.sip_trunk_id
            print(f"Inbound trunk created: {trunk_id}")
        else:
            trunk_id = trunks.items[0].sip_trunk_id
            print(f"Inbound trunk already exists: {trunk_id}")

        # Create (or recreate) one dispatch rule per extension.
        # Language is encoded in the room_prefix — the agent worker reads it.
        # SIP dispatch rules have no in-place update API, so to apply config
        # changes we delete any existing rule with the same name, then create it.
        existing_rules_by_name = {r.name: r.sip_dispatch_rule_id for r in rules.items}

        for ext in EXTENSIONS:
            existing_id = existing_rules_by_name.get(ext["rule_name"])
            if existing_id:
                await lk.sip.delete_sip_dispatch_rule(
                    api.DeleteSIPDispatchRuleRequest(sip_dispatch_rule_id=existing_id)
                )
                print(f"Rule '{ext['rule_name']}' existed — deleted {existing_id}")

            await lk.sip.create_sip_dispatch_rule(
                api.CreateSIPDispatchRuleRequest(
                    dispatch_rule=api.SIPDispatchRuleInfo(
                        name=ext["rule_name"],
                        trunk_ids=[trunk_id],
                        rule=api.SIPDispatchRule(
                            dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                                room_prefix=ext["room_prefix"],
                            )
                        ),
                        # Explicitly dispatch the named worker. Required because the
                        # worker registers with LIVEKIT_AGENT_NAME (my-agent), which
                        # puts it in explicit-dispatch mode and disables auto-join.
                        room_config=api.RoomConfiguration(
                            agents=[api.RoomAgentDispatch(agent_name="my-agent")],
                        ),
                        inbound_numbers=[ext["number"]],
                    )
                )
            )
            print(f"Rule '{ext['rule_name']}' created")

    finally:
        await lk.aclose()


asyncio.run(setup_sip())