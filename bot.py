import os
import json
import time
from datetime import datetime
from decimal import Decimal
from openai import OpenAI
from longbridge.openapi import (
    Config, QuoteContext, TradeContext,
    OrderType, OrderSide, TimeInForceType
)

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"].strip()
lb_config = Config.from_apikey_env()
quote_ctx = QuoteContext(lb_config)
trade_ctx = TradeContext(lb_config)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL_NAME = "deepseek/deepseek-chat"
RUN_MINUTES = int(os.environ.get("RUN_MINUTES", "18"))


def get_account_status() -> str:
    info = []
    for b in trade_ctx.account_balance():
        info.append(f"【账户资金】币种: {b.currency}, 总现金: {b.total_cash}, 净资产: {b.net_assets}")

    pos_resp = trade_ctx.stock_positions()
    positions = [
        f"代码: {p.symbol}, 持仓: {p.quantity}, 可卖: {p.available_quantity}, 成本价: {p.cost_price}"
        for ch in pos_resp.channels for p in ch.positions if int(p.quantity) > 0
    ]
    info.append("【当前持仓】\n" + ("\n".join(positions) if positions else "当前空仓"))

    orders = trade_ctx.today_orders()
    order_list = [
        f"代码: {o.symbol}, 方向: {o.side}, 数量: {o.quantity}, 价格: {o.price}, 状态: {o.status}"
        for o in orders[:10]
    ]
    info.append("【今日订单记录】\n" + ("\n".join(order_list) if order_list else "今日暂无订单"))
    return "\n\n".join(info)


def get_us_stock_quote(symbols: list[str]) -> str:
    quotes = quote_ctx.quote(symbols)
    lines = []
    for q in quotes:
        last_p = float(q.last_done)
        open_p = float(q.open) if float(q.open) > 0 else last_p
        chg = ((last_p - open_p) / open_p) * 100
        lines.append(f"代码: {q.symbol}, 最新价: {last_p:.2f} USD, 今开: {open_p:.2f} USD, 日内涨跌: {chg:+.2f}%")
    return "\n".join(lines)


def place_paper_limit_order(symbol: str, side: str, quantity: int, price: float, reason: str) -> str:
    if quantity > 50 or quantity <= 0:
        return "拦截：单笔模拟下单股数需在 1 到 50 股之间。"
    order_side = OrderSide.Buy if side.lower() == "buy" else OrderSide.Sell
    resp = trade_ctx.submit_order(
        symbol=symbol,
        order_type=OrderType.LO,
        side=order_side,
        submitted_quantity=Decimal(str(quantity)),
        time_in_force=TimeInForceType.Day,
        submitted_price=Decimal(str(round(price, 2))),
        remark=f"GH_Bot: {reason[:30]}"
    )
    return f"模拟盘下单成功！[{side}] {symbol} {quantity}股 @ {price:.2f} USD, 订单号: {resp.order_id}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_account_status",
            "description": "查询模拟盘账户资金、当前持仓股票及成本价、以及今日已提交的订单记录",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_stock_quote",
            "description": "获取美股实时报价，如 ['NVDA.US', 'AAPL.US', 'TSLA.US', 'MSFT.US', 'META.US']",
            "parameters": {
                "type": "object",
                "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
                "required": ["symbols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_paper_limit_order",
            "description": "在长桥模拟盘提交买入(Buy)或卖出(Sell)限价单",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["Buy", "Sell"]},
                    "quantity": {"type": "integer"},
                    "price": {"type": "number"},
                    "reason": {"type": "string"}
                },
                "required": ["symbol", "side", "quantity", "price", "reason"]
            }
        }
    }
]


def run_trading_cycle():
    print(f"\n================ 🕒 AI 巡检: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ================", flush=True)
    strategy_prompt = """
    你是我的长桥美股模拟盘量化交易员。请严格按以下顺序执行本轮巡检：
    1. 先调用 `get_account_status` 查看我当前的持仓、成本价以及今日已提交的订单。
    2. 调用 `get_us_stock_quote` 查询观察池 ['NVDA.US', 'AAPL.US', 'TSLA.US', 'MSFT.US', 'META.US'] 的最新行情。
    3. 【持仓止盈/止损检查】：
       - 若持仓盈利 >= +3%，立即调用 `place_paper_limit_order` 以最新价全部卖出（Sell）止盈；
       - 若持仓亏损 <= -2%，立即调用 `place_paper_limit_order` 以最新价全部卖出（Sell）止损；
    4. 【新开仓检查（严禁重复下单）】：
       - 如果某只股票已经在【当前持仓】中，或者在【今日订单记录】里已经有买入挂单，绝对不要重复买入！
       - 如果我当前「持仓股票种类 + 今日未成交买单股票种类」合计少于 2 只，请从观察池剩下的股票里挑选今日日内表现最强的一只，以最新价买入（Buy）5 股。
    """
    messages = [{"role": "user", "content": strategy_prompt}]

    for _ in range(6):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            temperature=0.1
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print("\n🤖 AI 巡检报告：\n", msg.content, flush=True)
            break

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            print(f"⚙️ 调用工具: {fn_name}({args})", flush=True)

            if fn_name == "get_account_status":
                res = get_account_status()
            elif fn_name == "get_us_stock_quote":
                res = get_us_stock_quote(**args)
            elif fn_name == "place_paper_limit_order":
                res = place_paper_limit_order(**args)
            else:
                res = "未知工具"

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(res)})


start_time = time.time()
run_trading_cycle()

while (time.time() - start_time) < (RUN_MINUTES * 60):
    time.sleep(5)
    pos_resp = trade_ctx.stock_positions()
    cost_map = {
        p.symbol: float(p.cost_price)
        for ch in pos_resp.channels for p in ch.positions if int(p.quantity) > 0
    }
    if cost_map:
        quotes = quote_ctx.quote(list(cost_map.keys()))
        for q in quotes:
            last_p = float(q.last_done)
            cost_p = cost_map[q.symbol]
            pnl_pct = ((last_p - cost_p) / cost_p) * 100
            print(f"⚡ [5秒云端盯盘] {q.symbol} 现价:{last_p:.2f} USD, 成本:{cost_p:.2f} USD, 盈亏:{pnl_pct:+.2f}%", flush=True)
            if pnl_pct >= 3.0 or pnl_pct <= -2.0:
                print(f"🚨 触发止盈/止损线 ({pnl_pct:+.2f}%)，立即唤醒 AI！", flush=True)
                run_trading_cycle()
                break
