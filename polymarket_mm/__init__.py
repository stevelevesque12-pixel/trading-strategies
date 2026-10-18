"""
Paper-trading market maker for Polymarket's CLOB.

Reads live order books over the public market websocket, decides two-sided
quotes on the YES token of each selected market, and simulates fills against
the real trade/book feed. No keys, no wallet, no real orders. See
POLYMARKET.md for the model's assumptions and how to read the report.
"""
