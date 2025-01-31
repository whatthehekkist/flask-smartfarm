/*** external js for apps (e.g., Dash) in flask server ***/


// assets/scroll.js
/*
- layout.py의 고정 사이드바 메뉴의 ↑↓ 스크롤 이벤트가 inline JS로 동작을 하지 않음을 발견
- 현재 경로 assets/scroll.js 에서 따로 DOM 객체의 노드를 확인하면서 이벤트리스너에 추가해주어야 스크롤 이벤트가 정상적으로 작동함
*/

// dash-scroll-link 클래스에 클릭 이벤트 리스너 추가
function addScrollListeners() {
    const links = document.querySelectorAll('.dash-scroll-link');

    /**
    if (links.length === 0) {
        console.log("No links found with class 'dash-scroll-link'");
    } else {
        console.log(links.length + " links found with class 'dash-scroll-link'");
    }
    **/

    links.forEach(link => {
        // 이미 추가된 리스너가 있을 수 있으므로 중복 추가 방지
        if (!link.dataset.listenerAdded) {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                console.log("Scroll link clicked");
                const direction = this.getAttribute('href') === '#top' ? 0 : document.body.scrollHeight;

                window.scrollTo({
                    top: direction,
                    behavior: 'smooth'
                });
            });
            link.dataset.listenerAdded = 'true'; // 리스너가 추가되었음을 표시
        }
    });
}

// DOM 변경 감지를 위한 MutationObserver 설정
const observer = new MutationObserver((mutations) => {
    mutations.forEach(mutation => {
        if (mutation.type === 'childList') {
            addScrollListeners(); // DOM 변경 시 리스너 추가
        }
    });
});

// 감지할 DOM 요소 설정
observer.observe(document.body, {
    childList: true,
    subtree: true // 하위 요소도 감지
});

// 초기 DOM 로드 시 리스너 추가
document.addEventListener("DOMContentLoaded", function() {
    addScrollListeners();
});




