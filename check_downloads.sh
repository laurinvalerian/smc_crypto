#!/bin/bash
# Quick download status checker
# Usage: bash check_downloads.sh

echo "═══════════════════════════════════════════"
echo "  Download Status ($(date '+%H:%M:%S'))"
echo "═══════════════════════════════════════════"

# Check running processes
FOREX_PID=$(pgrep -f "forex_data_downloader" 2>/dev/null)
CRYPTO_PID=$(pgrep -f "utils.data_downloader" 2>/dev/null)

# Crypto
crypto_1m=$(ls /root/bot/data/crypto/*_1m.parquet 2>/dev/null | wc -l)
crypto_total=$(ls /root/bot/data/crypto/*.parquet 2>/dev/null | wc -l)
if [ -n "$CRYPTO_PID" ]; then
    echo "  Crypto:       ${crypto_1m}/100 coins (${crypto_total} files) 🔄 Running (PID $CRYPTO_PID)"
else
    echo "  Crypto:       ${crypto_1m}/100 coins (${crypto_total} files) $([ $crypto_1m -ge 100 ] && echo '✅ Done' || echo '⚠️ Stopped')"
fi

# Forex
forex_1m=$(ls /root/bot/data/forex/*_1m.parquet 2>/dev/null | wc -l)
forex_total=$(ls /root/bot/data/forex/*.parquet 2>/dev/null | wc -l)
if [ -n "$FOREX_PID" ]; then
    echo "  Forex+Commod: ${forex_1m}/32 instruments (${forex_total} files) 🔄 Running (PID $FOREX_PID)"
else
    echo "  Forex+Commod: ${forex_1m}/32 instruments (${forex_total} files) $([ $forex_1m -ge 32 ] && echo '✅ Done' || echo '⚠️ Stopped')"
fi

# Commodities (subset of forex download)
commod=$(ls /root/bot/data/commodities/*.parquet 2>/dev/null | wc -l)
echo "  Commodities:  ${commod} files $([ $commod -gt 0 ] && echo '✅' || echo '⏳ Pending (after forex)')"

# Stocks
stocks_1m=$(ls /root/bot/data/stocks/*_5m.parquet 2>/dev/null | wc -l)
stocks_total=$(ls /root/bot/data/stocks/*.parquet 2>/dev/null | wc -l)
echo "  Stocks:       ${stocks_1m}/50 stocks (${stocks_total} files) ✅ Done"

# Latest activity
echo ""
echo "  Latest files:"
echo "    Forex:  $(ls -t /root/bot/data/forex/*_1m.parquet 2>/dev/null | head -1 | xargs -I{} basename {} 2>/dev/null || echo 'none')"
echo "    Crypto: $(ls -t /root/bot/data/crypto/*_1m.parquet 2>/dev/null | head -1 | xargs -I{} basename {} 2>/dev/null || echo 'none')"
echo "═══════════════════════════════════════════"
