(function(){      
	'use strict';
	var articleBox = $("div.article_content");
	articleBox.removeAttr("style");
	var head_str = "";       
	var foot_str = "";   
	var older = document.body.innerHTML;       
	var title= document.getElementsByClassName('article-title-box')[0].innerHTML; 
	var main_body = document.getElementsByClassName('article_content')[0].innerHTML;
	document.body.innerHTML = head_str + title + main_body + foot_str;
	$("#mainBox").width("100%");
	document.getElementsByTagName('body')[0].style.zoom=0.8;     
	window.print();
	document.body.innerHTML = older;
	return false;
})();