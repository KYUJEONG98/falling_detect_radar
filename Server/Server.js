const { parseRadar } = require('./parse'); //parse.js 같은 폴더안에
const mysql = require('mysql'); // mysql 연결->
/*//////////////////////////////////////////////////*



     npm install mysql 설치해야됨     
     


*//////////////////////////////////////////////////

// MySQL 연결 생성
const connection = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: 'Password789!',
  database: 'mobiusdb'
});

// MySQL 연결
connection.connect();

// 이벤트 처리기 등록
connection.query("SELECT COUNT(*) as cnt FROM mobiusdb.cin", function(err, result) {
  if (err) throw err;
  let count = result[0].cnt;
  console.log(`총 ${count}개의 데이터 존재`);

  // 일정 간격(1초)으로 mobiusdb.cin의 가장 최신 con 값을 가져옴 
  setInterval(function() {
    connection.query("SELECT COUNT(*) as cnt FROM mobiusdb.cin", function(err, result) {
      if (err) throw err;
      let newCount = result[0].cnt; //db의 전체 레코드 수 
      if (newCount > count) { // 레코드 수가 갱신되었다는 것을 확인하고
        count = newCount; // 만약 추가되었다면 다시 레코드 수를 갱신-> 항상 최신 값을 더함
        
          // -> 최신 값 유자 ORDER BY ri DESC mobiusdb.cin의 id는 "ri"
          connection.query("SELECT con FROM mobiusdb.cin ORDER BY ri DESC LIMIT 1", function(err, result) {
          if (err) throw err;
          var data = JSON.stringify([result[0].con]);
          var data = data.substring(2,data.length-2);
          //console.log(`새로 추가된 데이터: ${data}`);

          var jsonObject = parseRadar(data); // parseRadar로....

          var pos = {
               x: jsonObject.message.content !== null ? jsonObject.message.content.pos_x : null,
               y: jsonObject.message.content !== null ? jsonObject.message.content.pos_y : null,
               z: jsonObject.message.content !== null ? jsonObject.message.content.pos_z : null
           };
           // bpm
           var bpm = jsonObject.message.content !== null ? jsonObject.message.content.bpm : null;
           // hbr
           var hbr = jsonObject.message.content !== null ? jsonObject.message.content.hbr : null;
           // energy
           var eng = jsonObject.message.content !== null ? jsonObject.message.content.energy : null;

          console.log(JSON.stringify(pos), bpm, hbr, eng);

          //console.log(`새로 추가된 데이터: ${jsonObject}`);
        });
      }
    });
  }, 1000); // 1초마다 감시
});
