"""
Redis stream manager for reliable message processing.

This module provides high-level Redis stream operations with consumer groups,
retry logic, message validation, and monitoring capabilities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import redis.asyncio as redis
from pydantic import BaseModel


@dataclass
class StreamMessage:
    """Represents a Redis stream message."""

    message_id: str
    fields: Dict[str, str]
    timestamp: datetime
    stream_name: str
    consumer_group: str


class MessageHandler(BaseModel):
    """Message handler configuration."""

    handler_name: str
    handler_func: Callable
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 5


class RetryPolicy(BaseModel):
    """Retry policy configuration."""

    max_retries: int = 3
    initial_delay_ms: int = 1000
    backoff_multiplier: float = 2.0
    max_delay_ms: int = 30000
    jitter_factor: float = 0.1


class StreamMetrics(BaseModel):
    """Stream processing metrics."""

    messages_processed: int = 0
    messages_failed: int = 0
    messages_retried: int = 0
    average_processing_time: float = 0.0
    last_message_time: Optional[datetime] = None
    consumer_lag: int = 0
    error_rate: float = 0.0


class RedisStreamManager:
    """
    Manages Redis streams with consumer groups and reliable message processing.

    Features:
    - Consumer group management
    - Reliable message processing with acknowledgments
    - Retry logic with exponential backoff
    - Message validation and schema checking
    - Performance monitoring and metrics
    - Dead letter queue for failed messages
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        stream_name: str,
        consumer_group: str,
        consumer_name: str,
        retry_policy: Optional[RetryPolicy] = None,
        dead_letter_stream: Optional[str] = None,
        enable_monitoring: bool = True,
    ):
        """
        Initialize the stream manager.

        Args:
            redis_client: Async Redis client
            stream_name: Name of the Redis stream
            consumer_group: Consumer group name
            consumer_name: Consumer name
            retry_policy: Retry policy configuration
            dead_letter_stream: Stream for failed messages
            enable_monitoring: Whether to enable metrics collection
        """
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.retry_policy = retry_policy or RetryPolicy()
        self.dead_letter_stream = dead_letter_stream or f"{stream_name}_dlq"
        self.enable_monitoring = enable_monitoring

        # Message handlers
        self.handlers: Dict[str, MessageHandler] = {}

        # Processing state
        self.is_processing = False
        self.metrics = StreamMetrics()

        # Monitoring
        self.logger = logging.getLogger(f"RedisStreamManager.{stream_name}")

        # Processing loop task
        self._processing_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the stream manager."""
        try:
            # Create consumer group if it doesn't exist
            await self._create_consumer_group()

            # Create dead letter stream if needed
            if self.dead_letter_stream:
                await self._create_dead_letter_stream()

            self.logger.info(f"Stream manager initialized: {self.stream_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize stream manager: {e}")
            raise

    async def _create_consumer_group(self) -> None:
        """Create consumer group for the stream."""
        try:
            await self.redis_client.xgroup_create(
                self.stream_name, self.consumer_group, id="0", mkstream=True
            )
            self.logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                self.logger.info(f"Consumer group {self.consumer_group} already exists")
            else:
                raise

    async def _create_dead_letter_stream(self) -> None:
        """Create dead letter stream for failed messages."""
        try:
            # Create empty stream
            await self.redis_client.xadd(self.dead_letter_stream, {"init": "true"})
            await self.redis_client.xdel(
                self.dead_letter_stream, "0-1"
            )  # Remove init message
            self.logger.info(f"Created dead letter stream: {self.dead_letter_stream}")
        except Exception as e:
            self.logger.error(f"Failed to create dead letter stream: {e}")

    def register_handler(self, handler: MessageHandler) -> None:
        """
        Register a message handler.

        Args:
            handler: Message handler configuration
        """
        self.handlers[handler.handler_name] = handler
        self.logger.info(f"Registered handler: {handler.handler_name}")

    def unregister_handler(self, handler_name: str) -> bool:
        """
        Unregister a message handler.

        Args:
            handler_name: Name of handler to unregister

        Returns:
            True if handler was unregistered
        """
        if handler_name in self.handlers:
            del self.handlers[handler_name]
            self.logger.info(f"Unregistered handler: {handler_name}")
            return True
        return False

    async def start_processing(self) -> None:
        """Start processing messages from the stream."""
        if self.is_processing:
            self.logger.warning("Already processing messages")
            return

        self.is_processing = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        self.logger.info(f"Started processing stream: {self.stream_name}")

    async def stop_processing(self) -> None:
        """Stop processing messages."""
        self.is_processing = False

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        self.logger.info(f"Stopped processing stream: {self.stream_name}")

    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while self.is_processing:
            try:
                # Read messages from stream
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream_name: ">"},
                    count=10,  # Process up to 10 messages at once
                    block=1000,  # 1 second timeout
                )

                if messages:
                    # Process messages concurrently
                    tasks = []
                    for stream, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            task = asyncio.create_task(
                                self._process_message(message_id, fields)
                            )
                            tasks.append(task)

                    # Wait for all messages to be processed
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, message_id: str, fields: Dict[str, str]) -> None:
        """Process a single message."""
        start_time = time.time()

        try:
            # Parse message
            message = StreamMessage(
                message_id=message_id,
                fields=fields,
                timestamp=datetime.now(),
                stream_name=self.stream_name,
                consumer_group=self.consumer_group,
            )

            # Determine handler based on message type
            handler_name = fields.get("handler", "default")
            handler = self.handlers.get(handler_name)

            if not handler:
                self.logger.warning(f"No handler found for: {handler_name}")
                await self._acknowledge_message(message_id)
                return

            # Validate message schema if provided
            if handler.input_schema:
                if not self._validate_message_schema(fields, handler.input_schema):
                    await self._handle_failed_message(
                        message, "Schema validation failed"
                    )
                    return

            # Process message with retry logic
            success = await self._process_with_retry(message, handler)

            if success:
                await self._acknowledge_message(message_id)
                self._update_metrics(True, time.time() - start_time)
            else:
                await self._handle_failed_message(
                    message, "Processing failed after retries"
                )
                self._update_metrics(False, time.time() - start_time)

        except Exception as e:
            self.logger.error(f"Failed to process message {message_id}: {e}")
            await self._handle_failed_message(
                StreamMessage(
                    message_id,
                    fields,
                    datetime.now(),
                    self.stream_name,
                    self.consumer_group,
                ),
                str(e),
            )
            self._update_metrics(False, time.time() - start_time)

    async def _process_with_retry(
        self, message: StreamMessage, handler: MessageHandler
    ) -> bool:
        """Process message with retry logic."""
        retry_count = 0
        delay = self.retry_policy.initial_delay_ms / 1000.0

        while retry_count <= self.retry_policy.max_retries:
            try:
                # Process message
                result = await asyncio.wait_for(
                    handler.handler_func(message), timeout=handler.timeout_seconds
                )

                # Validate output schema if provided
                if handler.output_schema and not self._validate_output_schema(
                    result, handler.output_schema
                ):
                    raise ValueError("Output schema validation failed")

                return True

            except asyncio.TimeoutError:
                self.logger.warning(f"Handler timeout for message {message.message_id}")
                retry_count += 1
            except Exception as e:
                self.logger.warning(
                    f"Handler error for message {message.message_id}: {e}"
                )
                retry_count += 1

            if retry_count <= self.retry_policy.max_retries:
                # Calculate delay with jitter
                jitter = (
                    delay
                    * self.retry_policy.jitter_factor
                    * (2 * (time.time() % 1) - 1)
                )
                actual_delay = min(
                    delay + jitter, self.retry_policy.max_delay_ms / 1000.0
                )

                self.logger.info(
                    f"Retrying message {message.message_id} in {actual_delay:.2f}s (attempt {retry_count})"
                )
                await asyncio.sleep(actual_delay)

                # Exponential backoff
                delay *= self.retry_policy.backoff_multiplier

        return False

    async def _acknowledge_message(self, message_id: str) -> None:
        """Acknowledge a processed message."""
        try:
            await self.redis_client.xack(
                self.stream_name, self.consumer_group, message_id
            )
        except Exception as e:
            self.logger.error(f"Failed to acknowledge message {message_id}: {e}")

    async def _handle_failed_message(self, message: StreamMessage, error: str) -> None:
        """Handle a failed message by sending to dead letter queue."""
        try:
            # Add to dead letter stream
            dead_letter_fields = {
                "original_message_id": message.message_id,
                "original_stream": message.stream_name,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "consumer_group": message.consumer_group,
                "consumer_name": self.consumer_name,
            }

            await self.redis_client.xadd(self.dead_letter_stream, dead_letter_fields)

            # Acknowledge original message to remove from pending
            await self._acknowledge_message(message.message_id)

            self.logger.warning(
                f"Message {message.message_id} sent to dead letter queue: {error}"
            )

        except Exception as e:
            self.logger.error(f"Failed to handle dead letter message: {e}")

    def _validate_message_schema(
        self, fields: Dict[str, str], schema: Dict[str, Any]
    ) -> bool:
        """Validate message fields against schema."""
        try:
            # Simple schema validation - can be enhanced with JSON Schema
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in fields:
                    return False

            # Check field types if specified
            field_types = schema.get("properties", {})
            for field_name, field_spec in field_types.items():
                if field_name in fields:
                    expected_type = field_spec.get("type")
                    if expected_type == "string" and not isinstance(
                        fields[field_name], str
                    ):
                        return False
                    elif expected_type == "number":
                        try:
                            float(fields[field_name])
                        except ValueError:
                            return False

            return True

        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            return False

    def _validate_output_schema(self, result: Any, schema: Dict[str, Any]) -> bool:
        """Validate handler output against schema."""
        try:
            # Simple output validation
            if schema.get("type") == "dict" and not isinstance(result, dict):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Output schema validation error: {e}")
            return False

    def _update_metrics(self, success: bool, processing_time: float) -> None:
        """Update processing metrics."""
        if success:
            self.metrics.messages_processed += 1
        else:
            self.metrics.messages_failed += 1

        # Update average processing time
        total_messages = self.metrics.messages_processed + self.metrics.messages_failed
        if total_messages > 0:
            self.metrics.average_processing_time = (
                self.metrics.average_processing_time * (total_messages - 1)
                + processing_time
            ) / total_messages

        self.metrics.last_message_time = datetime.now()

        # Calculate error rate
        if total_messages > 0:
            self.metrics.error_rate = self.metrics.messages_failed / total_messages

    async def send_message(
        self, fields: Dict[str, str], handler_name: str = "default"
    ) -> str:
        """
        Send a message to the stream.

        Args:
            fields: Message fields
            handler_name: Handler name for the message

        Returns:
            Message ID

        Raises:
            Exception: If there is an error sending the message.
        """
        try:
            # Add handler name to fields
            message_fields = fields.copy()
            message_fields["handler"] = handler_name
            message_fields["timestamp"] = datetime.now().isoformat()

            # Send to stream
            message_id = await self.redis_client.xadd(self.stream_name, message_fields)

            self.logger.debug(f"Sent message {message_id} to stream {self.stream_name}")
            return message_id

        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise

    async def get_pending_messages(self) -> List[StreamMessage]:
        """Get pending messages for this consumer."""
        try:
            pending = await self.redis_client.xpending(
                self.stream_name, self.consumer_group
            )

            messages = []
            for message_id, consumer, idle_time, delivery_count in pending:
                # Get message details
                message_data = await self.redis_client.xrange(
                    self.stream_name, message_id, message_id
                )

                if message_data:
                    message_id, fields = message_data[0]
                    messages.append(
                        StreamMessage(
                            message_id=message_id,
                            fields=fields,
                            timestamp=datetime.now()
                            - timedelta(milliseconds=idle_time),
                            stream_name=self.stream_name,
                            consumer_group=self.consumer_group,
                        )
                    )

            return messages

        except Exception as e:
            self.logger.error(f"Failed to get pending messages: {e}")
            return []

    async def get_stream_info(self) -> Dict[str, Any]:
        """Get information about the stream."""
        try:
            info = await self.redis_client.xinfo_stream(self.stream_name)

            # Get consumer group info
            groups = await self.redis_client.xinfo_groups(self.stream_name)

            # Get consumer info
            consumers = await self.redis_client.xinfo_consumers(
                self.stream_name, self.consumer_group
            )

            return {
                "stream_info": info,
                "consumer_groups": groups,
                "consumers": consumers,
                "metrics": self.metrics.model_dump(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get stream info: {e}")
            return {}

    def get_metrics(self) -> StreamMetrics:
        """Get current metrics."""
        return self.metrics

    async def reset_metrics(self) -> None:
        """Reset metrics."""
        self.metrics = StreamMetrics()
        self.logger.info("Metrics reset")

    async def cleanup_dead_letter_queue(self, max_age_hours: int = 24) -> int:
        """
        Clean up old messages from dead letter queue.

        Args:
            max_age_hours: Maximum age of messages to keep

        Returns:
            Number of messages removed
        """
        if not self.dead_letter_stream:
            return 0

        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)

            # Get messages older than cutoff
            old_messages = await self.redis_client.xrange(
                self.dead_letter_stream, "-", f"{cutoff_timestamp}-0"
            )

            if old_messages:
                message_ids = [msg_id for msg_id, _ in old_messages]
                await self.redis_client.xdel(self.dead_letter_stream, *message_ids)

                self.logger.info(f"Cleaned up {len(message_ids)} old messages from DLQ")
                return len(message_ids)

            return 0

        except Exception as e:
            self.logger.error(f"Failed to cleanup dead letter queue: {e}")
            return 0
