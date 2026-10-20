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


def get_real_usd_cash() -> tuple[float, float]:
    """获取真实折算后的可用美元现金（含融资购买力）与美元净资产"""
    usd_cash = 0.0
    usd_net = 0.0
    for b in trade_ctx.account_balance():
        rate = 7.82 if str(b.currency).upper() == "HKD" else 1.0
        usd_cash = float(b.total_cash) / rate
        usd_net = float(b.net_assets) / rate

        cash_infos = getattr(b, "cash_infos", [])
        if cash_infos:
            net_avail_usd = 0.0
            for ci in cash_infos:
                c_rate = 7.82 if str(ci.currency).upper() == "HKD" else 1.0
                net_avail_usd += float(ci.available_cash) / c_rate
            usd_cash = min(usd_cash, net_avail_usd)

        rem_finance = getattr(b, "remaining_finance_amount", None)
        if rem_finance is not None:
            usd_cash = min(usd_cash, float(rem_finance) / rate)

    return round(max(0.0, usd_cash), 2), round(usd_net, 2)


def get_account_status() -> str:
    """获取真实可用资金、当前持仓（含实时浮盈）与今日订单"""
    usd_cash, usd_net = get_real_usd_cash()
    info = [
        f"【账户真实美元资产】\n"
        f"当前最高可用美元购买力: {usd_cash:.2f} USD\n"
        f"账户折合美元总净资产: {usd_net:.2f} USD"
    ]

    pos_resp = trade_ctx.stock_positions()
    positions = []
    pos_symbols = [p.symbol for ch in pos_resp.channels for p in ch.positions if int(p.quantity) > 0]
    quote_map = {q.symbol: float(q.last_done) for q in quote_ctx.quote(pos_symbols)} if pos_symbols else {}

    for ch in pos_resp.channels:
        for p in ch.positions:
            qty = int(p.quantity)
            if qty > 0:
                cost = float(p.cost_price)
                cur_price = quote_map.get(p.symbol, cost)
                pnl_pct = ((cur_price - cost) / cost * 100) if cost > 0 else 0.0
                positions.append(
                    f"代码: {p.symbol}, 持仓股数: {qty}, 可卖股数: {p.available_quantity}, "
                    f"成本均价: {cost:.2f} USD, 现价: {cur_price:.2f} USD, 浮盈亏: {pnl_pct:+.2f}%"
                )
    info.append("【当前持仓】\n" + ("\n".join(positions) if positions else "当前空仓"))

    orders = trade_ctx.today_orders()
    order_list = []
    for o in orders[:15]:
        order_list.append(
            f"订单号: {o.order_id}, 代码: {o.symbol}, 方向: {o.side}, "
            f"股数: {o.quantity}, 委托价: {o.price}, 状态: {o.status}"
        )
    info.append("【今日订单记录】\n" + ("\n".join(order_list) if order_list else "今日暂无订单"))
    return "\n\n".join(info)


def get_us_stock_quote(symbols: list[str]) -> str:
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
            f"今开涨跌: {intra_chg:+.2f}%, 最高: {q.high}, 最低: {q.low}, 成交量: {q.volume}"
        )
    return "\n".join(lines)


def get_kline_trend(symbol: str, period: str = "15m") -> str:
    period_map = {
        "5m": Period.Min_5,
        "15m": Period.Min_15,
        "60m": Period.Min_60,
        "day": Period.Day
    }
    p_enum = period_map.get(period, Period.Min_15)
    try:
        klines = quote_ctx.candlesticks(symbol, p_enum, 20, AdjustType.NoAdjust)
    except Exception:
        klines = quote_ctx.candlesticks(symbol, p_enum, 20)
    if not klines:
        return f"{symbol} 暂无K线数据"
    closes = [float(k.close) for k in klines]
    ma5 = sum(closes[-5:]) / min(len(closes), 5)
    ma10 = sum(closes[-10:]) / min(len(closes), 10)
    ma20 = sum(closes[-20:]) / min(len(closes), 20)
    recent_bars = [
        f"收:{float(k.close):.2f}(高:{float(k.high):.2f}/低:{float(k.low):.2f}/量:{k.volume})"
        for k in klines[-8:]
    ]
    return (
        f"【{symbol} ({period}) K线趋势】最新价: {closes[-1]:.2f} USD | MA5: {ma5:.2f}, MA10: {ma10:.2f}, MA20: {ma20:.2f}\n"
        f"最近8根K线: " + " -> ".join(recent_bars)
    )


def place_paper_limit_order(symbol: str, side: str, quantity: int, price: float, reason: str) -> str:
    """提交限价单，解除人为仓位比例限制，仅保留券商真实购买力上限校准"""
    if quantity <= 0:
        return "拦截：下单股数必须大于 0。"
    try:
        order_side = OrderSide.Buy if side.lower() == "buy" else OrderSide.Sell
        final_qty = int(quantity)

        if order_side == OrderSide.Buy:
            usd_cash, _ = get_real_usd_cash()
            if usd_cash < 100:
                return f"拦截：当前剩余可用美元不足 ({usd_cash:.2f} USD)，无法继续买入。"

            # 仅根据真实可用购买力上限与长桥接口最大股数做防报错校准，不设人为分仓上限
            max_affordable_qty = int((usd_cash * 0.98) // price)
            if final_qty > max_affordable_qty:
                print(f"⚠️ 自动校准至最大可用资金股数: {final_qty} -> {max_affordable_qty} 股", flush=True)
                final_qty = max_affordable_qty

            try:
                est = trade_ctx.estimate_max_purchase_quantity(
                    symbol=symbol,
                    order_type=OrderType.LO,
                    side=OrderSide.Buy,
                    price=Decimal(str(round(price, 2)))
                )
                cash_max = int(getattr(est, "cash_max_qty", 0))
                margin_max = int(getattr(est, "margin_max_qty", 0))
                broker_max = max(cash_max, margin_max)

                if broker_max <= 0:
                    return f"拦截：长桥返回 {symbol} 当前可买股数为 0，请换其他标的或使用现金额度！"
                if final_qty > int(broker_max * 0.98):
                    print(f"⚠️ 触发券商最大购买力校准: {final_qty} -> {int(broker_max * 0.98)} 股", flush=True)
                    final_qty = int(broker_max * 0.98)
            except Exception as e:
                print(f"估算最大可买股数异常: {e}", flush=True)

        if final_qty <= 0:
            return "拦截：当前可用资金不足以买入 1 股该股票。"

        resp = trade_ctx.submit_order(
            symbol=symbol,
            order_type=OrderType.LO,
            side=order_side,
            submitted_quantity=Decimal(str(final_qty)),
            time_in_force=TimeInForceType.Day,
            submitted_price=Decimal(str(round(price, 2))),
            remark=f"MaxAlpha: {reason[:25]}"
        )
        time.sleep(1.5)
        for o in trade_ctx.today_orders():
            if o.order_id == resp.order_id:
                if "Reject" in str(o.status):
                    return f"下单被长桥拒绝（状态: {o.status}），请勿重复提交该笔订单！"
                return f"下单成功！[{side}] {symbol} {final_qty}股 @ {price:.2f} USD, 状态: {o.status}, 订单号: {resp.order_id}"

        return f"已提交订单！[{side}] {symbol} {final_qty}股 @ {price:.2f} USD, 订单号: {resp.order_id}"
    except Exception as e:
        return f"下单报错: {e}"


def cancel_pending_order(order_id: str) -> str:
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
            "description": "查询账户可用美元购买力、总净资产、当前持仓（含实时盈亏）及今日订单记录",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_stock_quote",
            "description": "批量获取任意美股或ETF的实时报价、涨跌幅与成交量（代码格式如 NVDA.US）",
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
            "description": "获取指定美股的多周期 K 线走势、成交量与 MA5/MA10/MA20 均线数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {
                        "type": "string",
                        "enum": ["5m", "15m", "60m", "day"],
                        "description": "K线周期：5m(超短线)、15m(日内波段)、60m(趋势)、day(日线)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_paper_limit_order",
            "description": "提交美股限价单（Buy买入 / Sell卖出）",
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
            "description": "根据 order_id 撤销未成交的挂单",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]
            }
        }
    }
]


def run_trading_cycle():
    print(f"\n================ 🚀 AI 极限收益全权操盘: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ================", flush=True)
    strategy_prompt = """
    你是顶级华尔街量化与日内交易专家，拥有该美股模拟账户的 100% 绝对控制权！
    唯一核心目标：**发挥你的全部交易智慧，不设人为条条框框，实现账户净资产的极限增长**。

    【你的自由权限与武器库】：
    1. 先调用 `get_account_status` 掌握当前购买力、持仓盈亏及挂单情况（如有未成交老挂单可先行撤销）。
    2. 你可以自由扫描高弹性核心池：
       - 高波动热门股：['NVDA.US', 'TSLA.US', 'AMD.US', 'PLTR.US', 'META.US', 'AVGO.US', 'COIN.US', 'MSTR.US', 'SMCI.US', 'APP.US']
       - 3倍杠杆多空ETF：['TQQQ.US', 'SQQQ.US', 'SOXL.US', 'SOXS.US', 'TSLL.US', 'NVDL.US']
       - 你也可以随时查询任何你认为处于风口的美股代码（格式：`代码.US`）。
    3. 灵活调用 `get_kline_trend` 结合 `5m`、`15m` 或 `day` K线及成交量变化，自主判断趋势强弱、突破真伪与反转信号。
    4. **完全自主的仓位与止盈止损策略**：
       - 不设固定的持仓数量限制，也不设固定的仓位比例限制！遇到极强单边主升浪时，允许你重仓甚至集中火力进攻最强龙头；震荡行情下也可灵活分仓或利用反向ETF（如 SQQQ.US、SOXS.US）做空获利。
       - 绝不要机械地因为微小的浮亏而割肉，也不要在主升浪刚启动时就过早止盈。根据均线支撑、量价背离和动能衰竭信号自主决定何时加仓、何时锁利、何时换股。
       - 唯一底线：不要同时持有同一标的的正反向杠杆ETF（如同时持有 TQQQ 和 SQQQ）。
       - 为确保限价单快速成交，买入限价可比现价高 0.10~0.20 USD，卖出限价可比现价低 0.10~0.20 USD。
    """
    messages = [{"role": "user", "content": strategy_prompt}]

    for _ in range(20):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            temperature=0.4
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print("\n🤖 AI 操盘决策报告：\n", msg.content, flush=True)
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
last_ai_run = time.time()
run_trading_cycle()

while (time.time() - start_time) < (RUN_MINUTES * 60):
    time.sleep(5)
    # 1. 每隔 4 分钟主动唤醒一次 AI 扫描全市场新机会（即使当前空仓也能捕捉暴涨股）
    if (time.time() - last_ai_run) >= 240:
        print("⏰ 到达 4 分钟定期巡检节点，唤醒 AI 扫描市场机会！", flush=True)
        run_trading_cycle()
        last_ai_run = time.time()
        continue

    # 2. 5秒高频盯盘：持仓出现剧烈波动时立即唤醒 AI 接管
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
            print(f"⚡ [5秒高频盯盘] {q.symbol} 现价:{last_p:.2f} USD, 成本:{cost_p:.2f} USD, 浮盈亏:{pnl_pct:+.2f}%", flush=True)
            # 当单票盈利突破 4% 或回撤达 -2.5%（且距离上次唤醒超过60秒防刷屏）时，唤醒 AI 自主决定是否让利润奔跑或止损
            if (pnl_pct >= 4.0 or pnl_pct <= -2.5) and (time.time() - last_ai_run) >= 60:
                print(f"🚨 持仓大幅波动 ({pnl_pct:+.2f}%)，立即唤醒 AI 评估趋势！", flush=True)
                run_trading_cycle()
                last_ai_run = time.time()
                break
