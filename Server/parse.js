exports.parseRadar = (data) => {
    // added : 레이더 센서 데이터 체크

    // 받은 데이터 
    // aa55    |    10  | 01020020c500003e | 2c050000 | 00000000 |  0300    | 010000  | cb63 | 55aa
    // 시작     |   CMD  |      Serial      | Sequence | Reserved |  length  | Message |  CRC | End

    let dataJson = {
        cmd: '',
        serial: '',
        sequence: '',
        reserved: '',
        length: '',
        message: '',
        crc: ''
    }
    let parsedJson = {
        cmd: '',
        serial: '',
        sequence: '',
        reserved: '',
        length: '',
        message: '',
        crc: ''
    };
    // let _buffer = Buffer.from(data);

    // parse buffer
    dataJson.cmd = data.substring(4, 6);
    dataJson.serial = data.substring(6, 22);
    dataJson.sequence = data.substring(22, 30);
    dataJson.reserved = data.substring(30, 38);
    dataJson.length = data.substring(38, 42);
    dataJson.message = data.substring(42, data.length - 8);
    dataJson.crc = data.substring(data.length - 8, data.length - 4);

    // 내용 분석
    // * cmd ------------------------------------------------------------------
    let parsedCMD = '';
    switch (dataJson.cmd) {
        case "00":
            parsedCMD = 'Response OK';
            //console.log('DEVICE ----------> HOST');
            break;
        case "01":
            parsedCMD = 'Response Error';
            //console.log('DEVICE ----------> HOST');
            break;
        case "10":
            parsedCMD = 'Response Result';
            //console.log('DEVICE ----------> HOST');
            break;
        case "11":
            parsedCMD = 'Response Control';
            //console.log('DEVICE <---------- HOST');
            break;
        case "12":
            parsedCMD = 'Response Status';
            //console.log('DEVICE ----------> HOST');
            break;
        case "21":
            parsedCMD = 'Wifi Control';
            //console.log('DEVICE <---------- HOST');
            break;
        case "22":
            parsedCMD = 'Wifi Status';
            //console.log('DEVICE ----------> HOST');
            break;
        case "30":
            parsedCMD = 'send .BIN';
            //console.log('RADAR <---------- Wifi');
            break;
        case "31":
            parsedCMD = 'FOTA Information';
            //console.log('DEVICE <---------- HOST');
            break;
    }
    parsedJson.cmd = parsedCMD;

    // * serial id ------------------------------------------------------------------
    parsedJson.serial = dataJson.serial;

    // * sequence ------------------------------------------------------------------
    let sequenceJSON = {
        type: '',
        number: ''
    };
    if (dataJson.cmd === '10') {
        sequenceJSON.type = 'RADAR FRAME_NUMBER';
        sequenceJSON.number = dataJson.sequence;
    } else if (dataJson.cmd === '11' || dataJson.cmd === '12') {
        sequenceJSON.type = 'Host나 Device의 SEQUENCE NUMBER';
        sequenceJSON.number = dataJson.sequence;
    } else if (dataJson.cmd === '00' || dataJson.cmd === '01') {
        sequenceJSON.type = '요청 packet의 SEQUENCE';
        sequenceJSON.number = dataJson.sequence;
    } else {
        console.log('sequence : wrong information !');
    }
    parsedJson.sequence = sequenceJSON;

    // * reserved ------------------------------------------------------------------
    parsedJson.reserved = dataJson.reserved;

    // * length ------------------------------------------------------------------
    parsedJson.length = parseByteToInt(dataJson.length, 2);

    // * message ------------------------------------------------------------------
    let msgJson = {
        name: '',
        content: ''
    };
    // Response OK or Error
    if (dataJson.cmd === '00' || dataJson.cmd === '01') {
        // response ok or error
        let res_code = dataJson.message.substring(0, 2);
        let sub_code = dataJson.message.substring(2, 4);
        let res_len = dataJson.message.substring(4, 8);
        let res_msg = dataJson.message.substring(8, dataJson.message.length);

        let msgStructure = {
            res_code: res_code,
            sub_code: sub_code,
            res_len: res_len,
            res_msg: res_msg
        };

        msgJson.name = 'Response OK or Error';
        msgJson.content = msgStructure;

        parsedJson.message = msgJson;
    }
    // Response Control
    else if (dataJson.cmd === '11') {
        let msgStructure = {
            res_msg: dataJson.message
        };

        msgJson.name = 'Response Control';
        msgJson.content = msgStructure;

        parsedJson.message = msgJson;
    }
    // Response Status
    else if (dataJson.cmd === '12') {
        let msgStructure = {
            res_msg: dataJson.message
        };

        msgJson.name = 'Response Status';
        msgJson.content = msgStructure;

        parsedJson.message = msgJson;
    }
    // Response Result
    else if (dataJson.cmd === '10') {
        // indoor
        let msgStructureIndoor = {
            target_id: '',
            pos_x: '',
            pos_y: '',
            pos_z: '',
            bpm: '',
            hbr: '',
            therm: '',
            rsv: '',
            energy: '',
            point: '',
            type: '',
            status: '',
            v1: '',
            v2: '',
            y1: '',
            y2: ''
        };
        // area scan
        let msgStructureArea = {
            area_id: '',
            pos_x: '',
            pos_y: '',
            pos_z: '',
            energy: '',
            point: '',
            type: '',
            range_1: '',
            azimuth_1: '',
            range_2: '',
            azimuth_2: '',
            reserved: ''
        };

        let msgID = dataJson.message.substring(0, 2);
        let msgLength = dataJson.message.substring(2, 6);
        let convertedLength = parseInt(msgLength, 16);
        let msgBody = dataJson.message.substring(6, convertedLength);
        // multiple messages
        if (convertedLength < dataJson.message.substring(6, dataJson.message.length).length) {
            // multiple messages
            // msg id = 1 | msg len = 2 | msg body = **
            // 2 + 4 + x
            let messageList = [];

            let loadedCount = 0;
            let pointer = 0;
            while (loadedCount > dataJson.message.length) {

                let msgID = dataJson.message.substring(pointer, pointer + 2);
                let msgLen = dataJson.message.substring(pointer + 2, pointer + 6);
                let convertedLen = parseInt(msgLen, 16);
                let msgBody = dataJson.message.substring(pointer + 6, pointer + 6 + convertedLen);

                if (msgID === '01') {
                    // save
                    let _msgIndoor = {
                        name: 'indoor',
                        content: msgStructureIndoor
                    };
                    // indoor
                    _msgIndoor.content.target_id = msgBody.substring(0, 2); // 1byte
                    // 1 byte parse HEX -> INT
                    // _msgIndoor.content.pos_x = msgBody.substring(2, 4);     // 1byte
                    _msgIndoor.content.pos_x = parseByteToInt(msgBody.substring(2, 4), 1);
                    // _msgIndoor.content.pos_y = msgBody.substring(4, 6);     // 1byte
                    _msgIndoor.content.pos_y = parseByteToInt(msgBody.substring(4, 6), 1);
                    // _msgIndoor.content.pos_z = msgBody.substring(6, 8);     // 1byte
                    _msgIndoor.content.pos_z = parseByteToInt(msgBody.substring(6, 8), 1);
                    // _msgIndoor.content.bpm = msgBody.substring(8, 12);      // 2byte
                    _msgIndoor.content.bpm = parseByteToInt(msgBody.substring(8, 12), 2);
                    // _msgIndoor.content.hbr = msgBody.substring(12, 16);     // 2byte
                    _msgIndoor.content.hbr = parseByteToInt(msgBody.substring(12, 16), 2);
                    // _msgIndoor.content.therm = msgBody.substring(16, 20);   // 2byte
                    _msgIndoor.content.therm = parseByteToInt(msgBody.substring(16, 20), 2);
                    _msgIndoor.content.rsv = msgBody.substring(20, 24);     // 2byte
                    // _msgIndoor.content.energy = msgBody.substring(24, 32);  // 4byte
                    _msgIndoor.content.energy = parseByteToInt(msgBody.substring(24, 32), 4);
                    // _msgIndoor.content.point = msgBody.substring(32, 34);   // 1byte
                    _msgIndoor.content.point = parseByteToInt(msgBody.substring(32, 34), 1);
                    // _msgIndoor.content.type = msgBody.substring(34, 36);    // 1byte
                    _msgIndoor.content.type = getType(msgBody.substring(34, 36));
                    _msgIndoor.content.status = msgBody.substring(36, 38);  // 1byte
                    _msgIndoor.content.v1 = msgBody.substring(38, 40);      // 1byte
                    _msgIndoor.content.v2 = msgBody.substring(40, 42);      // 1byte
                    _msgIndoor.content.y1 = msgBody.substring(42, 44);      // 1byte
                    _msgIndoor.content.y2 = msgBody.substring(44, 46);      // 1byte

                    // add into message list
                    messageList.push(_msgIndoor);
                    _msgIndoor = {};
                }
                else if (msgID === '02') {
                    // save
                    let _msgArea = {
                        name: 'area',
                        content: msgStructureArea
                    };
                    //area scan
                    _msgArea.content.area_id = msgBody.substring(0, 2);     // 1byte
                    // _msgArea.content.pos_x = msgBody.substring(2, 4);       // 1byte
                    _msgArea.content.pos_x = parseByteToInt(msgBody.substring(2, 4), 1);
                    // _msgArea.content.pos_y = msgBody.substring(4, 6);       // 1byte
                    _msgArea.content.pos_y = parseByteToInt(msgBody.substring(4, 6), 1);
                    // _msgArea.content.pos_z = msgBody.substring(6, 8);       // 1byte
                    _msgArea.content.pos_z = parseByteToInt(msgBody.substring(6, 8), 1);
                    // _msgArea.content.energy = msgBody.substring(8, 16);     // 4byte
                    _msgArea.content.energy = parseByteToInt(msgBody.substring(8, 16), 4);
                    // _msgArea.content.point = msgBody.substring(16, 18);     // 1byte
                    _msgArea.content.point = parseByteToInt(msgBody.substring(16, 18), 1);
                    // _msgArea.content.type = msgBody.substring(18, 20);      // 1byte
                    _msgArea.content.type = getType(msgBody.substring(18, 20));
                    _msgArea.content.range_1 = msgBody.substring(20, 22);   // 1byte
                    _msgArea.content.azimuth_1 = msgBody.substring(22, 24); // 1byte
                    _msgArea.content.range_2 = msgBody.substring(24, 26);   // 1byte
                    _msgArea.content.azimuth_2 = msgBody.substring(26, 28); // 1byte
                    _msgArea.content.reserved = msgBody.substring(28, 30);  // 1byte

                    // add into message list
                    messageList.push(_msgArea);
                    _msgArea = {};
                } else {
                    console.log('multi message : wrong information !');
                }
                // update pointer
                pointer = pointer + 6 + convertedLen;
                loadedCount = pointer;
            }
            // add to parsed json
            parsedJson.message = messageList;
        }
        else {
            // single message
            if (msgID === '01') {
                // save
                let _msgIndoor = {
                    name: 'indoor',
                    content: msgStructureIndoor
                };
                // indoor
                _msgIndoor.content.target_id = msgBody.substring(0, 2); // 1byte
                // 1 byte parse HEX -> INT
                // _msgIndoor.content.pos_x = msgBody.substring(2, 4);     // 1byte
                _msgIndoor.content.pos_x = parseByteToInt(msgBody.substring(2, 4), 1);
                // _msgIndoor.content.pos_y = msgBody.substring(4, 6);     // 1byte
                _msgIndoor.content.pos_y = parseByteToInt(msgBody.substring(4, 6), 1);
                // _msgIndoor.content.pos_z = msgBody.substring(6, 8);     // 1byte
                _msgIndoor.content.pos_z = parseByteToInt(msgBody.substring(6, 8), 1);
                // _msgIndoor.content.bpm = msgBody.substring(8, 12);      // 2byte
                _msgIndoor.content.bpm = parseByteToInt(msgBody.substring(8, 12), 2);
                // _msgIndoor.content.hbr = msgBody.substring(12, 16);     // 2byte
                _msgIndoor.content.hbr = parseByteToInt(msgBody.substring(12, 16), 2);
                // _msgIndoor.content.therm = msgBody.substring(16, 20);   // 2byte
                _msgIndoor.content.therm = parseByteToInt(msgBody.substring(16, 20), 2);
                _msgIndoor.content.rsv = msgBody.substring(20, 24);     // 2byte
                // _msgIndoor.content.energy = msgBody.substring(24, 32);  // 4byte
                _msgIndoor.content.energy = parseByteToInt(msgBody.substring(24, 32), 4);
                // _msgIndoor.content.point = msgBody.substring(32, 34);   // 1byte
                _msgIndoor.content.point = parseByteToInt(msgBody.substring(32, 34), 1);
                // _msgIndoor.content.type = msgBody.substring(34, 36);    // 1byte
                _msgIndoor.content.type = getType(msgBody.substring(34, 36));
                _msgIndoor.content.status = msgBody.substring(36, 38);  // 1byte
                _msgIndoor.content.v1 = msgBody.substring(38, 40);      // 1byte
                _msgIndoor.content.v2 = msgBody.substring(40, 42);      // 1byte
                _msgIndoor.content.y1 = msgBody.substring(42, 44);      // 1byte
                _msgIndoor.content.y2 = msgBody.substring(44, 46);      // 1byte


                // add into parsedJson
                parsedJson.message = _msgIndoor;

            } else if (msgID === '02') {
                // save
                let _msgArea = {
                    name: 'area',
                    content: msgStructureArea
                };
                //area scan
                _msgArea.content.area_id = msgBody.substring(0, 2);     // 1byte
                // _msgArea.content.pos_x = msgBody.substring(2, 4);       // 1byte
                _msgArea.content.pos_x = parseByteToInt(msgBody.substring(2, 4), 1);
                // _msgArea.content.pos_y = msgBody.substring(4, 6);       // 1byte
                _msgArea.content.pos_y = parseByteToInt(msgBody.substring(4, 6), 1);
                // _msgArea.content.pos_z = msgBody.substring(6, 8);       // 1byte
                _msgArea.content.pos_z = parseByteToInt(msgBody.substring(6, 8), 1);
                // _msgArea.content.energy = msgBody.substring(8, 16);     // 4byte
                _msgArea.content.energy = parseByteToInt(msgBody.substring(8, 16), 4);
                // _msgArea.content.point = msgBody.substring(16, 18);     // 1byte
                _msgArea.content.point = parseByteToInt(msgBody.substring(16, 18), 1);
                // _msgArea.content.type = msgBody.substring(18, 20);      // 1byte
                _msgArea.content.type = getType(msgBody.substring(18, 20));
                _msgArea.content.range_1 = msgBody.substring(20, 22);   // 1byte
                _msgArea.content.azimuth_1 = msgBody.substring(22, 24); // 1byte
                _msgArea.content.range_2 = msgBody.substring(24, 26);   // 1byte
                _msgArea.content.azimuth_2 = msgBody.substring(26, 28); // 1byte
                _msgArea.content.reserved = msgBody.substring(28, 30);  // 1byte

                // add into parsedJson
                parsedJson.message = _msgArea;
            } else {
                console.log('single message : wrong information !');
            }

        }
    }
    else {
        console.log('response result message : wrong information !');
    }
    // * crc ------------------------------------------------------------------
    parsedJson.crc = dataJson.crc;

    // check output
    //console.log(parsedJson);
    // TODO : DATABASE upload
    return parsedJson;
    // TODO : 받은 데이터 중 뭔지 모르겠는 값
    // 7b2263746e616d65223a226c6564222c22636f6e223a2268656c6c6f227d3c454f463e
}

function parseByteToInt(str, byte) {
    let result = 0;
    let littleEndian = [];
    // into little Endian
    for (let i = 0; i < byte; ++i) {
        littleEndian.push(str.substring(i * 2, (i * 2) + 2));
    }
    // change into Big Endian
    littleEndian.reverse();

    // in one string
    let toConvert = '';
    littleEndian.forEach((element) => {
        toConvert += element;
    });
    let hex = '0x' + toConvert;
    result = parseInt(hex, 16);

    return result;
}

function getType(str) {
    return str === '00' ? '동적형성' : str === '01' ? '고정물체' : '지정';
}