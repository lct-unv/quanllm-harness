"""Reference for the plugin API shipped by the main quanllm-harness package."""

from quanllm_harness.plugins import PluginManifest, PluginVerificationResult
from quanllm_harness.tools import Tool


class ExamplePlugin:
    manifest = PluginManifest(
        name="example_plugin",
        version="1.0.0",
        requires_harness=">=0.1.2,<0.2",
        permissions=("tools.register", "verifiers.register", "events.subscribe"),
    )

    def setup(self, context):
        context.tools.register(
            Tool(
                name="example_plugin.double",
                description="Double one integer.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                handler=lambda arguments: {"value": arguments["value"] * 2},
                limitations=("Integers only.",),
            )
        )
        context.verifiers.register(
            "audit",
            lambda verification: PluginVerificationResult(
                summary=f"Observed {len(verification.claims)} claims."
            ),
        )
        context.events.subscribe(lambda event: None)
        return lambda: None


plugin = ExamplePlugin()
