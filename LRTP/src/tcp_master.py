
import sys
import time
import logging
import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp

logger = modbus_tk.utils.create_logger("console")

_IPAddress = "127.0.0.1"
_Slave = 1
_DataLen = 8


if __name__ == "__main__":
    try:
        # 连接MODBUS TCP从机
        master = modbus_tcp.TcpMaster(host = _IPAddress)
        master.set_timeout(5.0)
        logger.info("connected")
        
        while True:
            # 读保持寄存器
            logger.info(master.execute(_Slave, cst.READ_HOLDING_REGISTERS, 0, _DataLen))
            time.sleep(1)
            
    except modbus_tk.modbus.ModbusError as e:
        logger.error("%s- Code=%d" % (e, e.get_exception_code()))
