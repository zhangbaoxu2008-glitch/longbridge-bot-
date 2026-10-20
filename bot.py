import os
import json
import time
from datetime import datetime
from decimal import Decimal
from openai import OpenAI
from longbridge.openapi import (
    Config, QuoteContext, TradeContext,
    OrderType, OrderSide, TimeInForceType, Period, AdjustType
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
    """获取模拟盘全部资金、购买力、当前持仓盈亏及今日订单（含订单号供撤单使用）"""
    info = []
    for b in trade_ctx.account_balance():
        info.append(
            f"【账户资产】币种: {b.currency}, 可用现金: {b.total_cash}, "
            f"最大购买力: { getattr(b, 'buy_power', b.total_cash) }, 净资产总额: {b.net_assets}"
        )

    pos_resp = trade_ctx.stock_positions()
    positions = []
    for ch in pos_resp.channels:
        for p in ch.positions:
            if int(p.quantity) > 0:
                positions.append(
                    f"代码: {p.symbol}, 持仓股数: {p.quantity}, 可卖股数: {p.available_quantity}, 成本均价: {p.cost_price}"
                )
    info.append("【当前持仓】\n" + ("\n".join(positions) if positions else "当前空仓"))

    orders = trade_ctx.today_orders()
    order_list = []
    for o in orders[:15]:
        order_list.append(
            f"订单号: {o.order_id}, 代码: {o.symbol}, 方向: {o.side}, "
            f"委托股数: {o.quantity}, 委托价: {o.price}, 状态: {o.status}"
        )
    info.append("【今日订单记录】\n" + ("\n".join(order_list) if order_list else "今日暂无订单"))
    return "\n\n".join(info)


def get_us_stock_quote(symbols: list[str]) -> str:
    """批量获取美股实时报价、日内涨跌幅、振幅与成交量"""
    quotes = quote_ctx.quote(symbols)
    lines = []
    for q in quotes:
        last_p = float(q.last_done)
        open_p = float(q.open) if float(q.open) > 0 else last_p
        prev_c = float(getattr(q, "prev_close", open_p)) or open_p
        day_chg = ((last_p - prev_c) / prev_c) * 100
        intra_chg = ((last_p - open_p) / open_p) * 100
        lines.append(
            f"代码: {q.symbol}, 最新价: {last_p:.2f} USD, 昨收涨跌: {day_chg:+.2f}%, "
            f"今开涨跌: {intra_chg:+.2f}%, 今日最高: {q.high}, 今日最低: {q.low}, 成交量: {q.volume}"
        )
    return "\n".join(lines)


def get_kline_trend(symbol: str) -> str:
    """获取单只股票最近 15 分钟 K 线序列与短期均线趋势，辅助判断买点与卖点"""
    try:
        klines = quote_ctx.candlesticks(symbol, Period.Min_15, 12, AdjustType.NoAdjust)
    except Exception:
        klines = quote_ctx.candlesticks(symbol, Period.Min_15, 12)
    if not klines:
        return f"{symbol} 暂无K线数据"
    closes = [float(k.close) for k in klines]
    ma5 = sum(closes[-5:]) / min(len(closes), 5)
    ma10 = sum(closes[-10:]) / min(len(closes), 10)
    recent_bars = [
        f"收盘:{float(k.close):.2f}(最高:{float(k.high):.2f}/最低:{float(k.low):.2f}/量:{k.volume})"
        for k in klines[-5:]
    ]
    return (
        f"【{symbol} 15分钟K线趋势】最新价: {closes[-1]:.2f}, MA5均线: {ma5:.2f}, MA10均线: {ma10:.2f}\n"
        f"最近5根K线: " + " -> ".join(recent_bars)
    )


def place_paper_limit_order(symbol: str, side: str, quantity: int, price: float, reason: str) -> str:
    """提交美股模拟盘限价单，已解除股数上限，由 AI 根据资金自主决定仓位大小"""
    if quantity <= 0:
        return "拦截：下单股数必须大于 0。"
    try:
        order_side = OrderSide.Buy if side.lower() == "buy" else OrderSide.Sell
        resp = trade_ctx.submit_order(
            symbol=symbol,
            order_type=OrderType.LO,
            side=order_side,
            submitted_quantity=Decimal(str(int(quantity))),
            time_in_force=TimeInForceType.Day,
            submitted_price=Decimal(str(round(price, 2))),
            remark=f"FullAI: {reason[:28]}"
        )
        return f"下单成功！[{side}] {symbol} {quantity}股 @ {price:.2f} USD, 订单号: {resp.order_id}"
    except Exception as e:
        return f"下单失败（请检查可用资金或可卖股数）: {e}"


def cancel_pending_order(order_id: str) -> str:
    """撤销尚未成交的挂单以释放资金或重新改价下单"""
    try:
        trade_ctx.cancel_order(order_id)
        return f"撤单成功！已撤销订单号: {order_id}"
    except Exception as e:
        return f"撤单失败: {e}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_account_status",
            "description": "查询模拟盘账户全部可用现金、净资产、当前持仓股数与成本价、以及今日订单记录",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_stock_quote",
            "description": "批量获取美股实时报价与涨跌幅，可传入任意美股代码列表",
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
            "name": "get_kline_trend",
            "description": "获取指定美股的 15 分钟 K 线走势与 MA5/MA10 均线数据",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_paper_limit_order",
            "description": "提交美股限价单（Buy买入 / Sell卖出），股数 quantity 由你根据账户现金自主计算",
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
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_pending_order",
            "description": "根据 order_id 撤销未成交的挂单，释放购买力或可卖持仓",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]
            }
        }
    }
]


def run_trading_cycle():
    print(f"\n================ 🚀 AI 全权基金经理巡检: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ================", flush=True)
    strategy_prompt = """
    你现在拥有我长桥美股模拟账户的 100% 最高全权托管权限！你的唯一目标是：充分利用账户资金，实现账户净资产（net_assets）最大化！

    请按以下步骤自主决策并执行交易：
    1. 调用 `get_account_status` 查清当前有多少可用现金（total_cash）、已有持仓及未成交挂单。
       - 如果有长时间未成交（如 Status 为 New 或 WaitToNew 但价格已偏离现价）的挂单，可调用 `cancel_pending_order` 撤单释放资金。
    2. 调用 `get_us_stock_quote` 扫描全市场高弹性标的池：
       - 核心科技与高弹性牛股：['NVDA.US', 'TSLA.US', 'AAPL.US', 'AMD.US', 'PLTR.US', 'META.US', 'MSFT.US', 'AVGO.US', 'COIN.US', 'MSTR.US']
       - 3倍杠杆多空ETF（大盘涨买多，大盘跌买空）：['TQQQ.US', 'SQQQ.US', 'SOXL.US', 'SOXS.US']
    3. 对你最看好的 2~3 只标的或当前持仓标的，调用 `get_kline_trend` 查看 15 分钟 K 线与 MA5/MA10 均线趋势。
    4. 【全权仓位管理与主动换仓规则】：
       - 对于已有持仓：若短线趋势走弱、跌破均线或亏损达 -1.5% 以上，果断卖出（Sell）止损换股；若盈利丰厚且出现滞涨，果断卖出止盈；若趋势极强（最新价站稳 MA5 之上且放量突破），允许继续持有甚至加仓！
       - 对于可用现金：不要让大量现金闲置！请将账户 80%~95% 的资金分配到当前日内动能最强、K线处于上升趋势的 2~4 只标的上。
       - 【股数计算方法】：根据你想分配给该股票的美元金额除以当前最新价，计算出整数股数 `quantity`（例如打算买 20000 美元、股价 350 美元，就买 57 股，不再受 5 股限制！）。为了确保限价单快速成交，买入限价可设在最新价或比最新价高 0.05~0.10 USD，卖出限价可设在最新价或比最新价低 0.05~0.10 USD。
    5. 执行完所有买卖操作后，输出你本轮的持仓结构、资金利用率及操盘逻辑。
    """
    messages = [{"role": "user", "content": strategy_prompt}]

    for _ in range(10):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            temperature=0.2
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print("\n🤖 AI 基金经理操盘报告：\n", msg.content, flush=True)
            break

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            print(f"⚙️ 执行指令: {fn_name}({args})", flush=True)

            if fn_name == "get_account_status":
                res = get_account_status()
            elif fn_name == "get_us_stock_quote":
                res = get_us_stock_quote(**args)
            elif fn_name == "get_kline_trend":
                res = get_kline_trend(**args)
            elif fn_name == "place_paper_limit_order":
                res = place_paper_limit_order(**args)
            elif fn_name == "cancel_pending_order":
                res = cancel_pending_order(**args)
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
            if pnl_pct >= 2.5 or pnl_pct <= -1.5:
                print(f"🚨 异动触发 ({pnl_pct:+.2f}%)，立即唤醒 AI 基金经理决策！", flush=True)
                run_trading_cycle()
                break
