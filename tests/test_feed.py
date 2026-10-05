import asyncio

import bot.data.feed as feed_mod
from bot.data.feed import MarketDataFeed

SYM = "BTC/USDT:USDT"


class FakeWS:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    async def watch_tickers(self, symbols):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("Connection timeout")
        return {SYM: {"last": 200.0, "timestamp": 2}}


class FakeREST:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    async def fetch_tickers(self, symbols):
        self.calls += 1
        if self.fail:
            raise ConnectionError("exchange down")
        return {SYM: {"last": 100.0, "timestamp": 1}}


def make_feed(cfg, ws, rest):
    f = MarketDataFeed.__new__(MarketDataFeed)
    f.cfg, f.ws, f.rest = cfg.exchange, ws, rest
    cfg.exchange.rest_poll_seconds = 0.001
    return f


async def collect(f, n, errors):
    async def on_error(exc, where):
        errors.append(where)

    out = []
    async for sym, price, ts in f.stream_prices(lambda: [SYM], on_error, lambda: None):
        out.append(price)
        if len(out) >= n:
            break
    return out


def test_websocket_blips_are_bridged_by_rest_and_never_hit_kill_switch(cfg, monkeypatch):
    monkeypatch.setattr(feed_mod, "WS_RETRY_SECONDS", 0.01)
    ws, rest = FakeWS(fail_times=3), FakeREST()
    errors: list[str] = []

    async def run():
        first = await collect(f, 3, errors)          # three WS failures, each bridged by REST
        await asyncio.sleep(0.02)                     # past the retry delay
        later = await collect(f, 3, errors)
        return first, later

    f = make_feed(cfg, ws, rest)
    first, later = asyncio.run(run())
    assert first == [100.0, 100.0, 100.0]             # no gap in prices
    assert 200.0 in later                              # WebSocket was retried and restored
    assert errors == []                                # kill-switch counter untouched


def test_no_price_source_counts_toward_kill_switch(cfg):
    errors: list[str] = []
    f = make_feed(cfg, None, FakeREST(fail=True))

    async def run():
        task = asyncio.create_task(collect(f, 1, errors))
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(run())
    assert errors and all("no working price source" in e for e in errors)
